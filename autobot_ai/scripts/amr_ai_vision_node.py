#!/usr/bin/env python3

#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # initializes CUDA driver
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Point, Quaternion

class AIVisionNode:
    def __init__(self):
        rospy.init_node('amr_ai_vision_node', anonymous=True)
        rospy.loginfo("Initializing optimized Vision AI node with TensorRT...")
        
        self.costmap_resolution = float(rospy.get_param('~costmap_resolution', 0.05))
        self.engine_path = rospy.get_param('~engine_path', "/home/aravind/autobot_ai_ws/src/autobot_ai/models/Try it out/fast_scnn_5class_fp16.trt")

        self.bridge = CvBridge()
        self.color_map = np.array([
            [128, 64, 128],    # road (0)
            [244, 35, 232],    # sidewalk (1)
            [70, 70, 70],      # building (2)
            [107, 142, 35],    # vegetation (3)
            [70, 130, 180]     # sky (4)
        ], dtype=np.uint8)

        self.load_engine()
        self.setup_ros_communication()

        rospy.loginfo("TensorRT optimized node ready")

    def load_engine(self):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        rospy.loginfo(f"Number of bindings: {self.engine.num_bindings}")
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            is_input = self.engine.binding_is_input(i)
            shape = self.engine.get_binding_shape(i)
            rospy.loginfo(f"Binding {i}: name={name}, input={is_input}, shape={shape}")

        # Check if input shape is dynamic (-1 dimension)
        input_shape = self.engine.get_binding_shape(0)
        if -1 in input_shape:
            # You must set binding shape before inference for dynamic shape engine
            # Replace these with your expected model input size (height, width)
            expected_height = 512  
            expected_width = 512
            self.context.set_binding_shape(0, (1, 3, expected_height, expected_width))
            rospy.loginfo(f"Set dynamic input binding shape to (1,3,{expected_height},{expected_width})")

        # After setting shape, get actual binding shapes from context
        self.input_shape = self.context.get_binding_shape(0)
        self.output_shape = self.context.get_binding_shape(1)

        self.batch_size = 1
        self.input_size = trt.volume(self.input_shape) * self.batch_size
        self.output_size = trt.volume(self.output_shape) * self.batch_size

        # Allocate host and device buffers
        self.host_input = cuda.pagelocked_empty(self.input_size, np.float32)
        self.host_output = cuda.pagelocked_empty(self.output_size, np.float32)

        self.device_input = cuda.mem_alloc(self.host_input.nbytes)
        self.device_output = cuda.mem_alloc(self.host_output.nbytes)

        self.stream = cuda.Stream()

        # Normalization params
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std_inv = np.array([1/0.229, 1/0.224, 1/0.225], dtype=np.float32).reshape(1, 1, 3)

        rospy.loginfo(f"Loaded TensorRT engine with input shape {self.input_shape} and output shape {self.output_shape}")

    def setup_ros_communication(self):
        self.segmentation_pub = rospy.Publisher('segmentation_output', Image, queue_size=1, tcp_nodelay=True)
        self.colored_seg_pub = rospy.Publisher('colored_segmentation', Image, queue_size=1, tcp_nodelay=True)
        self.costmap_pub = rospy.Publisher('/semantic_costmap', OccupancyGrid, queue_size=1, tcp_nodelay=True)
        
        self.image_sub = rospy.Subscriber(
            '/camera/color/image_raw', 
            Image, 
            self.image_callback, 
            queue_size=1,
            buff_size=8*1024*1024
        )

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            cv_image = cv2.flip(cv_image, 1)
            pred_mask = self.process_image(cv_image)
            if pred_mask is not None:
                self.publish_results(cv_image, pred_mask)
        except Exception as e:
            rospy.logerr(f"Processing error: {str(e)}")

    def process_image(self, image):
        try:
            orig_height, orig_width = image.shape[:2]

            target_height, target_width = self.input_shape[2], self.input_shape[3]

            # If dynamic input shape, set it again before inference
            if -1 in self.engine.get_binding_shape(0):
                self.context.set_binding_shape(0, (1, 3, target_height, target_width))

            # Resize and normalize
            resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            normalized = (resized_rgb - self.mean) * self.std_inv

            # HWC -> CHW and make contiguous
            input_data = np.ascontiguousarray(normalized.transpose(2, 0, 1), dtype=np.float32).flatten()

            # Copy to host input buffer
            np.copyto(self.host_input, input_data)

            # Transfer input to device
            cuda.memcpy_htod_async(self.device_input, self.host_input, self.stream)

            # Run inference
            bindings = [int(self.device_input), int(self.device_output)]
            self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)

            # Transfer predictions back
            cuda.memcpy_dtoh_async(self.host_output, self.device_output, self.stream)
            self.stream.synchronize()

            # Reshape output and get argmax per pixel
            output = self.host_output.reshape(self.output_shape)
            pred_mask = np.argmax(output[0], axis=0).astype(np.uint8)

            # Resize mask back to original size
            pred_mask = cv2.resize(pred_mask, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)

            return pred_mask

        except Exception as e:
            rospy.logerr(f"TensorRT inference failed: {str(e)}")
            return None

    def publish_results(self, original_image, pred_mask):
        try:
            colored_seg = self.color_map[pred_mask]
            mono_mask = (pred_mask * 63).astype(np.uint8)
            self.colored_seg_pub.publish(self.bridge.cv2_to_imgmsg(colored_seg, "bgr8"))
            self.segmentation_pub.publish(self.bridge.cv2_to_imgmsg(mono_mask, "mono8"))

            costmap = self.create_costmap(pred_mask)
            if costmap:
                self.costmap_pub.publish(costmap)

        except Exception as e:
            rospy.logerr(f"Publishing error: {str(e)}")

    def create_costmap(self, segmentation_mask):
        try:
            costmap = OccupancyGrid()
            costmap.header.stamp = rospy.Time.now()
            costmap.header.frame_id = 'base_link'

            size_x = int(20 / self.costmap_resolution)
            size_y = int(20 / self.costmap_resolution)

            costmap.info.resolution = float(self.costmap_resolution)
            costmap.info.width = int(size_x)
            costmap.info.height = int(size_y)

            costmap.info.origin = Pose(
                Point(float(-10.0), float(-10.0), float(0)),
                Quaternion(float(0), float(0), float(0), float(1))
            )

            if segmentation_mask.dtype != np.uint8:
                segmentation_mask = segmentation_mask.astype(np.uint8)

            rotated_mask = cv2.rotate(segmentation_mask, cv2.ROTATE_90_CLOCKWISE)
            resized_mask = cv2.resize(rotated_mask, (size_x, size_y), interpolation=cv2.INTER_NEAREST)

            costmap_data = np.full((size_y, size_x), -1, dtype=np.int8)

            class_to_cost = {
                0: 0,
                1: 50,
                2: 100,
                3: 100,
                4: 0
            }

            for cls, cost_val in class_to_cost.items():
                costmap_data[resized_mask == cls] = int(cost_val)

            kernel = np.ones((3, 3), np.uint8)
            obstacles = (costmap_data == 100).astype(np.uint8)
            inflated = cv2.dilate(obstacles, kernel, iterations=1)
            costmap_data[(inflated == 1) & (costmap_data < 100)] = 75

            costmap.data = [int(x) for x in costmap_data.ravel().tolist()]

            return costmap

        except Exception as e:
            rospy.logerr(f"Costmap creation error: {str(e)}")
            return None

if __name__ == '__main__':
    try:
        node = AIVisionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass




# import rospy
# import numpy as np
# import cv2
# import onnxruntime as ort
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# from nav_msgs.msg import OccupancyGrid
# from geometry_msgs.msg import Pose, Point, Quaternion

# class AIVisionNode:
#     def __init__(self):
#         rospy.init_node('amr_ai_vision_node', anonymous=True)
#         rospy.loginfo("Initializing optimized Vision AI node...")
        
#         # Configuration with proper type casting
#         self.costmap_resolution = float(rospy.get_param('~costmap_resolution', 0.05))
#         self.model_path = rospy.get_param('~model_path', "/home/aravind/autobot_ai_ws/src/autobot_ai/models/Try it out/fast_scnn_5class.onnx")

#         # Initialize with proper types
#         self.bridge = CvBridge()
#         self.color_map = np.array([
#             [128, 64, 128],    # road (0)
#             [244, 35, 232],    # sidewalk (1)
#             [70, 70, 70],      # building (2) - Added missing class
#             [107, 142, 35],    # vegetation (3)
#             [70, 130, 180]     # sky (4)
#         ], dtype=np.uint8)
        
#         # Model setup
#         self.initialize_model()
        
#         # ROS communication
#         self.setup_ros_communication()
        
#         rospy.loginfo("Optimized node ready")

#     def initialize_model(self):
#         try:
#             so = ort.SessionOptions()
#             so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
#             so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
#             so.intra_op_num_threads = 4
#             so.inter_op_num_threads = 1
            
#             providers = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
            
#             self.session = ort.InferenceSession(
#                 self.model_path,
#                 sess_options=so,
#                 providers=providers
#             )
            
#             self.input_name = self.session.get_inputs()[0].name
#             self.input_shape = self.session.get_inputs()[0].shape
#             self.output_name = self.session.get_outputs()[0].name
            
#             # Normalization constants with proper types
#             self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
#             self.std_inv = np.array([1/0.229, 1/0.224, 1/0.225], dtype=np.float32).reshape(1, 1, 3)
            
#             self.target_size = (self.input_shape[3], self.input_shape[2])  # width, height
            
#             rospy.loginfo(f"Model loaded for {providers[0]} with input shape {self.input_shape}")
            
#         except Exception as e:
#             rospy.logerr(f"Model initialization failed: {str(e)}")
#             rospy.signal_shutdown("Critical failure")

#     def setup_ros_communication(self):
#         self.segmentation_pub = rospy.Publisher('segmentation_output', Image, queue_size=1, tcp_nodelay=True)
#         self.colored_seg_pub = rospy.Publisher('colored_segmentation', Image, queue_size=1, tcp_nodelay=True)
#         self.costmap_pub = rospy.Publisher('/semantic_costmap', OccupancyGrid, queue_size=1, tcp_nodelay=True)
        
#         self.image_sub = rospy.Subscriber(
#             '/camera/color/image_raw', 
#             Image, 
#             self.image_callback, 
#             queue_size=1,
#             buff_size=8*1024*1024
#         )

#     def image_callback(self, msg):
#         try:
#             # Convert and process image
#             cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
#             cv_image = cv2.flip(cv_image, 1)  # horizontal flip

#             pred_mask = self.process_image(cv_image)
            
#             if pred_mask is not None:
#                 self.publish_results(cv_image, pred_mask)
            
#         except Exception as e:
#             rospy.logerr(f"Processing error: {str(e)}")

#     def process_image(self, image):
#         try:
#             # 1. Validate input image
#             if not isinstance(image, np.ndarray):
#                 raise ValueError("Input must be a numpy array")
#             if image.dtype != np.uint8:
#                 image = image.astype(np.uint8)

#             # 2. Get original dimensions
#             orig_height, orig_width = image.shape[:2]
            
#             # 3. Handle dynamic dimensions in model input shape
#             model_input_shape = self.session.get_inputs()[0].shape
            
#             # If shape contains strings (dynamic dimensions), use original image dimensions
#             if any(isinstance(dim, str) for dim in model_input_shape):
#                 target_height, target_width = orig_height, orig_width
#             else:
#                 target_height = int(model_input_shape[2])
#                 target_width = int(model_input_shape[3])

#             # 4. Resize and convert color space
#             resized = cv2.resize(
#                 image, 
#                 (target_width, target_height),
#                 interpolation=cv2.INTER_LINEAR
#             )
            
#             # 5. Convert to RGB and normalize
#             resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
#             normalized = (resized_rgb / 255.0 - self.mean) * self.std_inv

#             # 6. Prepare input tensor
#             input_tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1))[np.newaxis]
            
#             # 7. Run inference
#             outputs = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
            
#             # 8. Process output and resize back
#             pred_mask = np.argmax(outputs[0], axis=0).astype(np.uint8)
#             return cv2.resize(
#                 pred_mask,
#                 (orig_width, orig_height),
#                 interpolation=cv2.INTER_NEAREST
#             )
            
#         except Exception as e:
#             rospy.logerr(f"Image processing failed: {str(e)}")
#             import traceback
#             rospy.logerr(traceback.format_exc())
#             return None
    

#     def publish_results(self, original_image, pred_mask):
#         try:
#             # Color the segmentation mask
#             colored_seg = self.color_map[pred_mask]

#             # Mono mask scaled to 0-255 (5 classes: 0-4 -> 0, 63, 127, 191, 255)
#             mono_mask = (pred_mask * 63).astype(np.uint8)
            
#             # Publish results
#             self.colored_seg_pub.publish(self.bridge.cv2_to_imgmsg(colored_seg, "bgr8"))
#             self.segmentation_pub.publish(self.bridge.cv2_to_imgmsg(mono_mask, "mono8"))
            
#             # Create and publish costmap
#             costmap = self.create_costmap(pred_mask)
#             if costmap is not None:
#                 self.costmap_pub.publish(costmap)
                
#         except Exception as e:
#             rospy.logerr(f"Publishing error: {str(e)}")

#     def create_costmap(self, segmentation_mask):
#         try:
#             costmap = OccupancyGrid()
#             costmap.header.stamp = rospy.Time.now()
#             costmap.header.frame_id = 'base_link'
            
#             size_x = int(20 / self.costmap_resolution)
#             size_y = int(20 / self.costmap_resolution)
            
#             costmap.info.resolution = float(self.costmap_resolution)
#             costmap.info.width = int(size_x)
#             costmap.info.height = int(size_y)
            
#             # Set origin (centered)
#             costmap.info.origin = Pose(
#                 Point(float(-10.0), float(-10.0), float(0)),
#                 Quaternion(float(0), float(0), float(0), float(1))
#             )
            
#             # Ensure mask is proper type
#             if segmentation_mask.dtype != np.uint8:
#                 segmentation_mask = segmentation_mask.astype(np.uint8)
            
#             # Rotate and resize
#             rotated_mask = cv2.rotate(segmentation_mask, cv2.ROTATE_90_CLOCKWISE)
#             resized_mask = cv2.resize(rotated_mask, (size_x, size_y), 
#                                     interpolation=cv2.INTER_NEAREST)
            
#             # Initialize costmap data
#             costmap_data = np.full((size_y, size_x), -1, dtype=np.int8)  # -1 = unknown
            
#             # Class to cost mapping (must match your 5 classes)
#             class_to_cost = {
#                 0: 0,    # road - free space
#                 1: 50,   # sidewalk - traversable but higher cost
#                 2: 100,  # building - obstacle
#                 3: 100,  # vegetation - obstacle
#                 4: 0     # sky - free space
#             }
            
#             # Apply costs
#             for cls, cost_val in class_to_cost.items():
#                 costmap_data[resized_mask == cls] = int(cost_val)
            
#             # Inflate obstacles slightly
#             kernel = np.ones((3, 3), np.uint8)
#             obstacles = (costmap_data == 100).astype(np.uint8)
#             inflated = cv2.dilate(obstacles, kernel, iterations=1)
#             costmap_data[(inflated == 1) & (costmap_data < 100)] = 75  # inflated cost
            
#             # Convert to list of integers
#             costmap.data = [int(x) for x in costmap_data.ravel().tolist()]
            
#             return costmap
            
#         except Exception as e:
#             rospy.logerr(f"Costmap creation error: {str(e)}")
#             return None

# if __name__ == '__main__':
#     try:
#         node = AIVisionNode()
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         pass




# #Optimized ONNX
# import rospy
# import numpy as np
# import cv2
# import onnxruntime as ort
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# from nav_msgs.msg import OccupancyGrid
# from geometry_msgs.msg import Pose, Point, Quaternion

# class AIVisionNode:
#     def __init__(self):
#         rospy.init_node('amr_ai_vision_node', anonymous=True)
#         rospy.loginfo("Initializing optimized Vision AI node...")
        
#         # Configuration
#         self.costmap_resolution = rospy.get_param('~costmap_resolution', 0.05)
#         # self.model_path = rospy.get_param('~model_path', "/home/aravind/autobot_ai_ws/src/autobot_ai/models/fast_scnn.onnx")
#         self.model_path = rospy.get_param('~model_path', "/home/aravind/autobot_ai_ws/src/autobot_ai/models/Try it out/fast_scnn_fp32.onnx")

#         # At the top of your __init__():
#         self.costmap_resolution = rospy.get_param('~costmap_resolution', 0.05)
#         # Performance-critical initialization
#         self.bridge = CvBridge()
#         self.color_map = np.array([
#             [128, 64, 128],    # road
#             [244, 35, 232],    # sidewalk
#             [107, 142, 35],    # vegetation
#             [70, 130, 180]     # sky
#         ], dtype=np.uint8)
        
#         # Model setup with maximum optimization
#         self.initialize_model()
        
#         # ROS communication
#         self.setup_ros_communication()
        
#         rospy.loginfo("Optimized node ready")

#     def initialize_model(self):
#         """Ultra-optimized model initialization"""
#         try:
#             # Configure ONNX Runtime for maximum performance
#             so = ort.SessionOptions()
#             so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
#             so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
#             so.intra_op_num_threads = 4  # Match CPU cores
#             so.inter_op_num_threads = 1
            
#             # Use CUDA if available, otherwise CPU with optimizations
#             providers = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
            
#             self.session = ort.InferenceSession(
#                 self.model_path,
#                 sess_options=so,
#                 providers=providers
#             )
            
#             # Cache model metadata
#             self.input_name = self.session.get_inputs()[0].name
#             self.input_shape = self.session.get_inputs()[0].shape
#             self.output_name = self.session.get_outputs()[0].name
            
#             # Pre-allocate normalization constants with correct types
#             self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
#             self.std_inv = np.array([1/0.229, 1/0.224, 1/0.225], dtype=np.float32).reshape(1, 1, 3)
            
#             # Pre-compute target size
#             self.target_size = (self.input_shape[3], self.input_shape[2])  # (width, height)
            
#             rospy.loginfo(f"Model loaded for {providers[0]} with input shape {self.input_shape}")
            
#         except Exception as e:
#             rospy.logerr(f"Model initialization failed: {str(e)}")
#             rospy.signal_shutdown("Critical failure")

#     def setup_ros_communication(self):
#         """Optimized ROS communication setup"""
#         # Publishers with buffering
#         self.segmentation_pub = rospy.Publisher('segmentation_output', Image, queue_size=1, tcp_nodelay=True)
#         self.colored_seg_pub = rospy.Publisher('colored_segmentation', Image, queue_size=1, tcp_nodelay=True)
#         self.costmap_pub = rospy.Publisher('/semantic_costmap', OccupancyGrid, queue_size=1, tcp_nodelay=True)
        
#         # Subscriber with buffered transport
#         self.image_sub = rospy.Subscriber(
#             '/camera/color/image_raw', 
#             Image, 
#             self.image_callback, 
#             queue_size=1,
#             buff_size=8*1024*1024  # 8MB buffer
#         )

#     def image_callback(self, msg):
#         """Optimized image processing pipeline"""
#         try:
#             # Convert ROS image to numpy array
#             cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
#             cv_image = cv2.flip(cv_image, 1)  # 1 = horizontal flip

#             # Process image (all heavy operations happen here)
#             pred_mask = self.process_image(cv_image)
            
#             # Publish results in parallel
#             self.publish_results(cv_image, pred_mask)
            
#         except Exception as e:
#             rospy.logerr(f"Processing error: {str(e)}")

#     def process_image(self, image):
#         """Fixed and optimized image processing pipeline"""
#         # Combined BGR2RGB and resize in single operation
#         resized = cv2.cvtColor(
#             cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR),
#             cv2.COLOR_BGR2RGB
#         ).astype(np.float32)
        
#         # Optimized normalization using pre-computed values
#         # Fixed: Using multiply instead of divide for std
#         normalized = (resized / 255.0 - self.mean) * self.std_inv
        
#         # Efficient CHW conversion with contiguous memory
#         input_tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1))[np.newaxis]
        
#         # Run inference
#         outputs = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        
#         # Get predictions and resize in one step
#         return cv2.resize(
#             np.argmax(outputs[0], axis=0).astype(np.uint8),
#             (image.shape[1], image.shape[0]),
#             interpolation=cv2.INTER_NEAREST
#         )

#     def publish_results(self, original_image, pred_mask):
#         """Parallel result publishing"""
#         # Colored segmentation (vectorized operation)
#         colored_seg = self.color_map[pred_mask]
        
#         # Mono mask (optimized calculation)
#         mono_mask = (pred_mask * 85).astype(np.uint8)  # 255/3 ≈ 85
        
#         # Publish in parallel
#         self.colored_seg_pub.publish(
#             self.bridge.cv2_to_imgmsg(colored_seg, "bgr8"))
#         self.segmentation_pub.publish(
#             self.bridge.cv2_to_imgmsg(mono_mask, "mono8"))
#         self.costmap_pub.publish(
#             self.create_costmap(pred_mask))



#     def create_costmap(self, segmentation_mask):
        
#         """Generate optimized semantic costmap aligned with navigation needs"""
#         costmap = OccupancyGrid()
#         costmap.header.stamp = rospy.Time.now()
#         costmap.header.frame_id = 'base_link'  # Must match move_base config

#         # Set map dimensions to match global_costmap
#         size_x = int(20 / self.costmap_resolution)  # 20m width
#         size_y = int(20 / self.costmap_resolution)  # 20m height
#         costmap.info.resolution = self.costmap_resolution
#         costmap.info.width = size_x
#         costmap.info.height = size_y

#         # CRITICAL FIX - Set origin to (-10, -15)
#         # costmap.info.origin = Pose(
#         #     Point(-10.0, -17.0, 0),
#         #     Quaternion(0, 0, 0, 1)
#         # )

        
#         #  CRITICAL FIX - Set origin to (-10, -15)
#         costmap.info.origin = Pose(
#             Point(-10.0, -10.0, 0),
#             Quaternion(0, 0, 0, 1)
#         )

#             # Rotate the segmentation mask 90° clockwise to match robot orientation
#         rotated_mask = cv2.rotate(segmentation_mask, cv2.ROTATE_90_CLOCKWISE)
        
#         # Rest of your existing costmap generation code...
#         resized = cv2.resize(
#             rotated_mask,  # Use rotated mask instead of original
#             (size_x, size_y), 
#             interpolation=cv2.INTER_NEAREST
#         )



#         # # Resize segmentation mask to match costmap size
#         # resized = cv2.resize(
#         #     segmentation_mask, 
#         #     (size_x, size_y), 
#         #     interpolation=cv2.INTER_NEAREST
#         # )

#         # Vectorized cost assignment
#         costmap_data = np.full_like(resized, -1, dtype=np.int8)  # Default: unknown
#         costmap_data[resized == 0] = 0     # Free space (road)
#         costmap_data[resized == 1] = 50    # Medium cost (sidewalk)
#         costmap_data[resized == 2] = 100   # Lethal obstacle (vegetation)

#         # Inflate obstacles
#         obstacle_mask = (costmap_data == 100)
#         if np.any(obstacle_mask):
#             inflated = cv2.dilate(
#                 obstacle_mask.astype(np.uint8),
#                 np.ones((3, 3), np.uint8),
#                 iterations=1
#             )
#             costmap_data[(inflated == 1) & ~obstacle_mask] = 75  # Inflated cost

#         # Serialize to ROS format
#         costmap.data = costmap_data.ravel().tolist()

#         rospy.logdebug(f"Published costmap: {size_x}x{size_y} @ {self.costmap_resolution}m/cell")
        
#         return costmap

    



# if __name__ == '__main__':
#     try:
#         node = AIVisionNode()
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         pass





#With ONNX - perfect
# import rospy
# import numpy as np
# import cv2
# import onnxruntime as ort
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# from nav_msgs.msg import OccupancyGrid
# from geometry_msgs.msg import Pose, Point, Quaternion

# class AIVisionNode:
#     def __init__(self):
#         rospy.init_node('amr_ai_vision_node', anonymous=True)
#         rospy.loginfo("Initializing optimized Vision AI node...")
        
#         # Configuration
#         self.costmap_resolution = rospy.get_param('~costmap_resolution', 0.05)
#         self.model_path = rospy.get_param('~model_path', "/home/aravind/autobot_ai_ws/src/autobot_ai/models/fast_scnn.onnx")
#         # At the top of your __init__():
#         self.costmap_resolution = rospy.get_param('~costmap_resolution', 0.05)
#         # Performance-critical initialization
#         self.bridge = CvBridge()
#         self.color_map = np.array([
#             [128, 64, 128],    # road
#             [244, 35, 232],    # sidewalk
#             [107, 142, 35],    # vegetation
#             [70, 130, 180]     # sky
#         ], dtype=np.uint8)
        
#         # Model setup with maximum optimization
#         self.initialize_model()
        
#         # ROS communication
#         self.setup_ros_communication()
        
#         rospy.loginfo("Optimized node ready")

#     def initialize_model(self):
#         """Ultra-optimized model initialization"""
#         try:
#             # Configure ONNX Runtime for maximum performance
#             so = ort.SessionOptions()
#             so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
#             so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
#             so.intra_op_num_threads = 4  # Match CPU cores
#             so.inter_op_num_threads = 1
            
#             # Use CUDA if available, otherwise CPU with optimizations
#             providers = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
            
#             self.session = ort.InferenceSession(
#                 self.model_path,
#                 sess_options=so,
#                 providers=providers
#             )
            
#             # Cache model metadata
#             self.input_name = self.session.get_inputs()[0].name
#             self.input_shape = self.session.get_inputs()[0].shape
#             self.output_name = self.session.get_outputs()[0].name
            
#             # Pre-allocate normalization constants with correct types
#             self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
#             self.std_inv = np.array([1/0.229, 1/0.224, 1/0.225], dtype=np.float32).reshape(1, 1, 3)
            
#             # Pre-compute target size
#             self.target_size = (self.input_shape[3], self.input_shape[2])  # (width, height)
            
#             rospy.loginfo(f"Model loaded for {providers[0]} with input shape {self.input_shape}")
            
#         except Exception as e:
#             rospy.logerr(f"Model initialization failed: {str(e)}")
#             rospy.signal_shutdown("Critical failure")

#     def setup_ros_communication(self):
#         """Optimized ROS communication setup"""
#         # Publishers with buffering
#         self.segmentation_pub = rospy.Publisher('segmentation_output', Image, queue_size=1, tcp_nodelay=True)
#         self.colored_seg_pub = rospy.Publisher('colored_segmentation', Image, queue_size=1, tcp_nodelay=True)
#         self.costmap_pub = rospy.Publisher('/semantic_costmap', OccupancyGrid, queue_size=1, tcp_nodelay=True)
        
#         # Subscriber with buffered transport
#         self.image_sub = rospy.Subscriber(
#             '/camera/color/image_raw', 
#             Image, 
#             self.image_callback, 
#             queue_size=1,
#             buff_size=8*1024*1024  # 8MB buffer
#         )

#     def image_callback(self, msg):
#         """Optimized image processing pipeline"""
#         try:
#             # Convert ROS image to numpy array
#             cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
#             cv_image = cv2.flip(cv_image, 1)  # 1 = horizontal flip

#             # Process image (all heavy operations happen here)
#             pred_mask = self.process_image(cv_image)
            
#             # Publish results in parallel
#             self.publish_results(cv_image, pred_mask)
            
#         except Exception as e:
#             rospy.logerr(f"Processing error: {str(e)}")

#     def process_image(self, image):
#         """Fixed and optimized image processing pipeline"""
#         # Combined BGR2RGB and resize in single operation
#         resized = cv2.cvtColor(
#             cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR),
#             cv2.COLOR_BGR2RGB
#         ).astype(np.float32)
        
#         # Optimized normalization using pre-computed values
#         # Fixed: Using multiply instead of divide for std
#         normalized = (resized / 255.0 - self.mean) * self.std_inv
        
#         # Efficient CHW conversion with contiguous memory
#         input_tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1))[np.newaxis]
        
#         # Run inference
#         outputs = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        
#         # Get predictions and resize in one step
#         return cv2.resize(
#             np.argmax(outputs[0], axis=0).astype(np.uint8),
#             (image.shape[1], image.shape[0]),
#             interpolation=cv2.INTER_NEAREST
#         )

#     def publish_results(self, original_image, pred_mask):
#         """Parallel result publishing"""
#         # Colored segmentation (vectorized operation)
#         colored_seg = self.color_map[pred_mask]
        
#         # Mono mask (optimized calculation)
#         mono_mask = (pred_mask * 85).astype(np.uint8)  # 255/3 ≈ 85
        
#         # Publish in parallel
#         self.colored_seg_pub.publish(
#             self.bridge.cv2_to_imgmsg(colored_seg, "bgr8"))
#         self.segmentation_pub.publish(
#             self.bridge.cv2_to_imgmsg(mono_mask, "mono8"))
#         self.costmap_pub.publish(
#             self.create_costmap(pred_mask))

#     # def create_costmap(self, segmentation_mask):
#     #     """Hyper-optimized costmap generation"""
#     #     costmap = OccupancyGrid()
#     #     costmap.header.stamp = rospy.Time.now()
#     #     costmap.header.frame_id = 'odom'
        
#     #     # In create_costmap():
#     #     size_x = int(50 / self.costmap_resolution)  # For 50m map
#     #     size_y = int(50 / self.costmap_resolution)
#     #     costmap.info.width = size_x
#     #     costmap.info.height = size_y

#     #     # # Calculate map parameters
#     #     # size = int(10 / self.costmap_resolution)
#     #     # costmap.info.resolution = self.costmap_resolution
#     #     # costmap.info.width = costmap.info.height = size
#     #     # costmap.info.origin = Pose(
#     #     #     Point(-(size * self.costmap_resolution)/2, 
#     #     #           -(size * self.costmap_resolution)/2, 0),
#     #     #     Quaternion(0, 0, 0, 1)
#     #     # )

#     #     costmap.info.origin = Pose(
#     #         Point(-10.0, -25.0, 0),  # Force the origin you want
#     #         Quaternion(0, 0, 0, 1)
#     #     )
        
#     #     # Resize and classify in one step
#     #     resized = cv2.resize(segmentation_mask, (size, size), interpolation=cv2.INTER_NEAREST)
        
#     #     resized = cv2.flip(resized, 1)  # Flip horizontally if needed




#     #     # Vectorized cost mapping
#     #     costmap_data = np.select(
#     #         [resized == 0, resized == 1, resized == 2],
#     #         [0, 50, 100],
#     #         default=-1
#     #     ).astype(np.int8)
        
#     #     # Optimized inflation
#     #     if np.any(costmap_data == 100):
#     #         inflated = cv2.dilate(
#     #             (costmap_data == 100).astype(np.uint8), 
#     #             np.ones((3,3), np.uint8), 
#     #             iterations=1
#     #         )
#     #         costmap_data[(inflated == 1) & (costmap_data != 100)] = 75
        
#     #     # Efficient serialization
#     #     costmap.data = costmap_data.ravel().tolist()
#     #     return costmap
        
#     def create_costmap(self, segmentation_mask):
#         """Fixed costmap generation with correct origin"""
#         costmap = OccupancyGrid()
#         costmap.header.stamp = rospy.Time.now()
#         costmap.header.frame_id = 'map'
        
#         # Set map dimensions to match global_costmap
#         size_x = int(20 / self.costmap_resolution)  # 50m width
#         size_y = int(20 / self.costmap_resolution)  # 50m height
        
#         costmap.info.resolution = self.costmap_resolution
#         costmap.info.width = size_x
#         costmap.info.height = size_y
        
#         # CRITICAL FIX - Set origin to (-10,-25)
#         # costmap.info.origin = Pose(
#         #     Point(-25.0, -10.0, 0),
#         #     Quaternion(0, 0, 0, 1)
#         # )
        
#         costmap.info.origin = Pose(
#             Point(-10.0, -15.0, 0),
#             Quaternion(0, 0, 0, 1)
#         )
        
#         # Process segmentation mask
#         resized = cv2.resize(segmentation_mask, (size_x, size_y), 
#                             interpolation=cv2.INTER_NEAREST)
        
#         # Vectorized cost mapping
#         costmap_data = np.select(
#             [resized == 0, resized == 1, resized == 2],
#             [0, 50, 100],  # Adjust these values as needed
#             default=-1
#         ).astype(np.int8)
        
#         # Apply inflation if needed
#         if np.any(costmap_data == 100):
#             kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
#             inflated = cv2.dilate(
#                 (costmap_data == 100).astype(np.uint8), 
#                 kernel, 
#                 iterations=1
#             )
#             costmap_data[(inflated == 1) & (costmap_data != 100)] = 75
        
#         costmap.data = costmap_data.ravel().tolist()
#         return costmap




# if __name__ == '__main__':
#     try:
#         node = AIVisionNode()
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         pass

#-----------------------------------------------------------------------------------------------
#Pytorch .pth weights
#----------------------------------------------------------------------------------------

# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# import rospy
# import numpy as np
# import cv2
# import torch
# import torchvision
# from torchvision import transforms
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# from nav_msgs.msg import OccupancyGrid
# from geometry_msgs.msg import Pose, Point, Quaternion
# from std_msgs.msg import Header

# from autobot_ai.algorithms.AI_vision import FastSCNN

# class AIVisionNode:
#     def __init__(self):
#         rospy.init_node('amr_ai_vision_node', anonymous=True)
#         rospy.loginfo("Vision AI node initialization started.")
        
#         self.bridge = CvBridge()
        
#         # Get parameters with defaults
#         self.costmap_resolution = rospy.get_param('~costmap_resolution', 0.05)
#         self.model_path = rospy.get_param('~model_path', 
#                                        "/home/aravind/autobot_ai_ws/src/autobot_ai/models/best_model_class4.pth")
#         self.costmap_size = rospy.get_param('~costmap_size', 20.0)  # meters
        
#         # Load model
#         try:
#             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#             rospy.loginfo(f"Using device: {self.device}")
            
#             self.model = FastSCNN()
#             state_dict = torch.load(self.model_path, map_location=self.device)
            
#             if 'model' in state_dict:
#                 state_dict = state_dict['model']
                
#             self.model.load_state_dict(state_dict, strict=False)
#             self.model.to(self.device)
#             self.model.eval()
            
#             rospy.loginfo("PyTorch model loaded successfully.")
#         except Exception as e:
#             rospy.logerr(f"Failed to load model: {str(e)}")
#             rospy.signal_shutdown("Model loading failed")
#             return

#         # Publishers with appropriate queue sizes
#         self.segmentation_pub = rospy.Publisher('segmentation_output', Image, queue_size=1)
#         self.colored_seg_pub = rospy.Publisher('colored_segmentation', Image, queue_size=1)
#         self.costmap_pub = rospy.Publisher('/semantic_costmap', OccupancyGrid, queue_size=1)

#         # Subscriber with buffering
#         self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback, queue_size=1, buff_size=2**24)

#         # Precompute transform
#         self.transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((512, 1024)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                std=[0.229, 0.224, 0.225])
#         ])

#         # Predefined color map
#         self.color_map = np.array([
#             [128, 64, 128],    # road
#             [244, 35, 232],    # sidewalk
#             [107, 142, 35],    # vegetation
#             [70, 130, 180],    # sky
#         ], dtype=np.uint8)

#         # Initialize costmap parameters
#         self.costmap_size_pixels = int(self.costmap_size / self.costmap_resolution)
#         rospy.loginfo(f"Costmap size: {self.costmap_size_pixels}x{self.costmap_size_pixels} pixels")

#     def create_colored_segmentation(self, pred_mask):
#         """Optimized color mapping using numpy indexing"""
#         return self.color_map[pred_mask]

#     def image_callback(self, msg):
#         if not hasattr(self, 'model'):
#             return
            
#         try:
#             # Convert ROS image to OpenCV
#             cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
#             cv_image = cv2.flip(cv_image, 1)  # Horizontal flip
            
#             # Process image and get prediction
#             pred_mask = self.process_image(cv_image)

#             # Publish colored segmentation
#             colored_seg = self.create_colored_segmentation(pred_mask)
#             self.colored_seg_pub.publish(self.bridge.cv2_to_imgmsg(colored_seg, "bgr8"))

#             # Publish mono mask
#             mono_mask = (pred_mask * (255 // 3)).astype(np.uint8)
#             self.segmentation_pub.publish(self.bridge.cv2_to_imgmsg(mono_mask, "mono8"))

#             # Create and publish costmap in a separate thread
#             costmap_msg = self.create_costmap(pred_mask)
#             if costmap_msg:
#                 self.costmap_pub.publish(costmap_msg)

#         except Exception as e:
#             rospy.logerr(f"Error processing image: {str(e)}")
#             rospy.logerr(f"Error traceback: {traceback.format_exc()}")

#     def process_image(self, image):
#         """Process image through the model"""
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image_tensor = self.transform(image).unsqueeze(0).to(self.device)

#         with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
#             main_out, _ = self.model(image_tensor)
#             return torch.argmax(main_out, dim=1).squeeze().cpu().numpy()

#     def create_costmap(self, segmentation_mask):
#         """Generate optimized semantic costmap"""
#         try:
#             # Start timing for performance monitoring
#             start_time = rospy.Time.now()
            
#             # Create empty costmap message
#             costmap = OccupancyGrid()
#             costmap.header.stamp = rospy.Time.now()
#             costmap.header.frame_id = 'odom'  # Must match your TF tree
            
#             # Set map parameters
#             costmap.info.resolution = self.costmap_resolution
#             costmap.info.width = self.costmap_size_pixels
#             costmap.info.height = self.costmap_size_pixels
            
#             # Set origin (adjust these values according to your robot's coordinate system)
#             costmap.info.origin = Pose(
#                 Point(-self.costmap_size/2, -self.costmap_size/2, 0),
#                 Quaternion(0, 0, 0, 1)
#             )

#             # Rotate and resize mask to match costmap orientation and size
#             rotated_mask = cv2.rotate(segmentation_mask, cv2.ROTATE_90_CLOCKWISE)
#             resized = cv2.resize(
#                 rotated_mask,
#                 (self.costmap_size_pixels, self.costmap_size_pixels),
#                 interpolation=cv2.INTER_NEAREST
#             )

#             # Initialize costmap data
#             costmap_data = np.full(resized.shape, -1, dtype=np.int8)  # -1 = unknown
            
#             # Class mappings to cost values
#             costmap_data[resized == 0] = 0    # Free space (road)
#             costmap_data[resized == 1] = 50    # Medium cost (sidewalk)
#             costmap_data[resized == 2] = 100   # Lethal obstacle (vegetation)
            
#             # Optimized inflation using OpenCV
#             if np.any(resized == 2):  # Only inflate if obstacles exist
#                 kernel = np.ones((3, 3), np.uint8)
#                 inflated = cv2.dilate(
#                     (resized == 2).astype(np.uint8),
#                     kernel,
#                     iterations=2
#                 )
#                 costmap_data[(inflated == 1) & (costmap_data != 100)] = 75  # Inflated cost

#             # Flatten and convert to list
#             costmap.data = costmap_data.ravel().tolist()
            
#             # Log processing time
#             processing_time = (rospy.Time.now() - start_time).to_sec()
#             rospy.logdebug(f"Costmap generation took {processing_time:.3f} seconds")
            
#             return costmap
            
#         except Exception as e:
#             rospy.logerr(f"Error in costmap generation: {str(e)}")
#             return None

# if __name__ == '__main__':
#     try:
#         node = AIVisionNode()
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         pass
#     except Exception as e:
#         rospy.logerr(f"Fatal error in main: {str(e)}")

        
        
# #ORIGINAL with Pytorch - Cuda/Cpu - model weights .pth
    
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# import rospy
# import numpy as np
# import cv2
# import torch
# import torchvision
# from torchvision import transforms
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# from nav_msgs.msg import OccupancyGrid
# from geometry_msgs.msg import Pose, Point, Quaternion
# from std_msgs.msg import Header

# from autobot_ai.algorithms.AI_vision import FastSCNN


# class AIVisionNode:
#     def __init__(self):
#         rospy.init_node('amr_ai_vision_node', anonymous=True)
#         rospy.loginfo("Vision AI node initialization started.")
        
#         self.bridge = CvBridge()
        
#         # Get parameters with defaults
#         self.costmap_resolution = rospy.get_param('~costmap_resolution', 0.05)
#         self.indoor_size = rospy.get_param('~indoor_size', 40)
#         self.outdoor_size = rospy.get_param('~outdoor_size', 100)
#         self.model_path = rospy.get_param('~model_path', 
#                                          "/home/aravind/autobot_ai_ws/src/autobot_ai/models/best_model_class4.pth")
        
#         # Load model
#         try:
#             # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#             self.device = torch.device('cuda')

#             rospy.loginfo(f"Using device: {self.device}")
            
#             self.model = FastSCNN()
#             state_dict = torch.load(self.model_path, map_location=self.device)
            
#             # Handle potential mismatch in model architecture
#             if 'model' in state_dict:
#                 state_dict = state_dict['model']
                
#             self.model.load_state_dict(state_dict, strict=False)
#             self.model.to(self.device)
#             self.model.eval()
            
#             rospy.loginfo("PyTorch model loaded successfully.")
#         except Exception as e:
#             rospy.logerr(f"Failed to load model: {str(e)}")
#             rospy.signal_shutdown("Model loading failed")
#             return

#         # Publishers
#         self.segmentation_pub = rospy.Publisher('segmentation_output', Image, queue_size=1)
#         self.colored_seg_pub = rospy.Publisher('colored_segmentation', Image, queue_size=1)
#         self.costmap_pub = rospy.Publisher('/semantic_costmap', OccupancyGrid, queue_size=1)

#         # Subscriber
#         self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback, queue_size=1)

#         # Precompute transform
#         self.transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((512, 1024)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#         ])

#     def class_id_to_color(self, class_id):
#         color_map = {
#             0: (128, 64, 128),    # road
#             1: (244, 35, 232),    # sidewalk
#             2: (107, 142, 35),    # vegetation
#             3: (70, 130, 180),    # sky
#         }
#         return color_map.get(class_id, (0, 0, 0))

#     def create_colored_segmentation(self, pred_mask):
#         h, w = pred_mask.shape
#         colored = np.zeros((h, w, 3), dtype=np.uint8)
#         for class_id in np.unique(pred_mask):
#             colored[pred_mask == class_id] = self.class_id_to_color(class_id)
#         return colored

#     def image_callback(self, msg):
#         if not hasattr(self, 'model'):
#             return
            
#         try:
#             cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
#             pred_mask = self.process_image(cv_image)

#             # Publish colored segmentation
#             colored_seg = self.create_colored_segmentation(pred_mask)
#             self.colored_seg_pub.publish(self.bridge.cv2_to_imgmsg(colored_seg, "bgr8"))

#             # Publish mono mask
#             mono_mask = (pred_mask * (255 // 3)).astype(np.uint8)
#             self.segmentation_pub.publish(self.bridge.cv2_to_imgmsg(mono_mask, "mono8"))

#             # Create and publish costmap
#             costmap_msg = self.create_costmap(pred_mask)
#             self.costmap_pub.publish(costmap_msg)

#         except Exception as e:
#             rospy.logerr(f"Error processing image: {str(e)}")

#     def process_image(self, image):
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image_tensor = self.transform(image).unsqueeze(0).to(self.device)

#         with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
#             main_out, _ = self.model(image_tensor)
#             pred_classes = torch.argmax(main_out, dim=1)
#             return pred_classes.squeeze().cpu().numpy()

   
#     #     return costmap
#     def create_costmap(self, segmentation_mask):
#         """Generate optimized semantic costmap aligned with navigation needs"""
#         costmap = OccupancyGrid()
#         costmap.header.stamp = rospy.Time.now()
#         costmap.header.frame_id = 'odom'  # Must match move_base config
        
#         # Dynamic sizing based on resolution (10m x 10m area)
#         size = int(10 / self.costmap_resolution)  # 200 cells for 0.05m resolution
#         costmap.info.resolution = self.costmap_resolution
#         costmap.info.width = size
#         costmap.info.height = size
        
#         # Center costmap at (0,0) in odom frame
#         costmap.info.origin.position.x = -(size * self.costmap_resolution)/2
#         costmap.info.origin.position.y = -(size * self.costmap_resolution)/2
#         costmap.info.origin.orientation.w = 1.0  # Neutral orientation

#         # More efficient resizing and cost mapping
#         resized = cv2.resize(
#             segmentation_mask, 
#             (size, size), 
#             interpolation=cv2.INTER_NEAREST
#         )
        
#         # Vectorized cost mapping (faster than np.select)
#         costmap_data = np.full_like(resized, -1, dtype=np.int8)  # Default unknown
#         costmap_data[resized == 0] = 0    # Free space (road)
#         costmap_data[resized == 1] = 50   # Medium cost (sidewalk)
#         costmap_data[resized == 2] = 100  # Lethal obstacle (vegetation)
        
#         # Optimized inflation
#         obstacle_mask = (costmap_data == 100)
#         if np.any(obstacle_mask):  # Only inflate if obstacles exist
#             inflated = cv2.dilate(
#                 obstacle_mask.astype(np.uint8), 
#                 np.ones((3,3), np.uint8), 
#                 iterations=1
#             )
#             costmap_data[(inflated == 1) & ~obstacle_mask] = 75  # Inflated cost
        
#         # Efficient data serialization
#         costmap.data = costmap_data.ravel().tolist()  # ravel() is faster than flatten()
        
#         rospy.logdebug(f"Published costmap: {size}x{size} @ {self.costmap_resolution}m/cell "
#                     f"({costmap.info.origin.position.x},{costmap.info.origin.position.y})")
#         return costmap

# if __name__ == '__main__':
#     try:
#         node = AIVisionNode()
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         pass




#-----------------------------------------------------------------------------------------------------------------------
#Tensor RT
#-------------------------------------------------

# import rospy
# import numpy as np
# import cv2
# import pycuda.driver as cuda
# import pycuda.autoinit
# import tensorrt as trt
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# from nav_msgs.msg import OccupancyGrid
# from geometry_msgs.msg import Pose, Point, Quaternion


# class TRTInference:
#     def __init__(self, engine_path):
#         self.logger = trt.Logger(trt.Logger.WARNING)
#         with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
#             self.engine = runtime.deserialize_cuda_engine(f.read())
#         self.context = self.engine.create_execution_context()

#         # Allocate buffers
#         self.input_shape = self.engine.get_binding_shape(0)
#         self.output_shape = self.engine.get_binding_shape(1)
#         self.input_dtype = trt.nptype(self.engine.get_binding_dtype(0))
#         self.output_dtype = trt.nptype(self.engine.get_binding_dtype(1))

#         self.input_host = cuda.pagelocked_empty(trt.volume(self.input_shape), self.input_dtype)
#         self.output_host = cuda.pagelocked_empty(trt.volume(self.output_shape), self.output_dtype)
#         self.input_device = cuda.mem_alloc(self.input_host.nbytes)
#         self.output_device = cuda.mem_alloc(self.output_host.nbytes)
#         self.bindings = [int(self.input_device), int(self.output_device)]

#         self.stream = cuda.Stream()

#     def infer(self, input_tensor):
#         np.copyto(self.input_host, input_tensor.ravel())
#         cuda.memcpy_htod_async(self.input_device, self.input_host, self.stream)
#         self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
#         cuda.memcpy_dtoh_async(self.output_host, self.output_device, self.stream)
#         self.stream.synchronize()
#         return self.output_host.reshape(self.output_shape)


# class AIVisionNode:
#     def __init__(self):
#         rospy.init_node('amr_ai_vision_node', anonymous=True)
#         rospy.loginfo("Initializing TensorRT-based Vision AI node...")

#         # Parameters
#         self.costmap_resolution = rospy.get_param('~costmap_resolution', 0.05)
#         self.model_path = rospy.get_param('~model_path', "/home/aravind/autobot_ai_ws/src/autobot_ai/models/Try it out/fast_scnn_fp16.trt")

#         self.bridge = CvBridge()
#         self.color_map = np.array([
#             [128, 64, 128],    # road
#             [244, 35, 232],    # sidewalk
#             [107, 142, 35],    # vegetation
#             [70, 130, 180]     # sky
#         ], dtype=np.uint8)

#         # Model
#         self.trt_model = TRTInference(self.model_path)
#         self.input_height = self.trt_model.input_shape[2]
#         self.input_width = self.trt_model.input_shape[3]

#         # Precomputed normalization
#         self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
#         self.std_inv = np.array([1/0.229, 1/0.224, 1/0.225], dtype=np.float32).reshape(1, 1, 3)

#         # ROS communication
#         self.segmentation_pub = rospy.Publisher('segmentation_output', Image, queue_size=1, tcp_nodelay=True)
#         self.colored_seg_pub = rospy.Publisher('colored_segmentation', Image, queue_size=1, tcp_nodelay=True)
#         self.costmap_pub = rospy.Publisher('/semantic_costmap', OccupancyGrid, queue_size=1, tcp_nodelay=True)

#         self.image_sub = rospy.Subscriber(
#             '/camera/color/image_raw',
#             Image,
#             self.image_callback,
#             queue_size=1,
#             buff_size=8 * 1024 * 1024
#         )

#         rospy.loginfo("TensorRT node ready")

#     def preprocess(self, image):
#         resized = cv2.cvtColor(
#             cv2.resize(image, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR),
#             cv2.COLOR_BGR2RGB
#         ).astype(np.float32)
#         normalized = (resized / 255.0 - self.mean) * self.std_inv
#         chw = normalized.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
#         return chw

#     def image_callback(self, msg):
#         try:
#             cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
#             input_tensor = self.preprocess(cv_image)
#             output = self.trt_model.infer(input_tensor)
#             pred = np.argmax(output[0], axis=0).astype(np.uint8)

#             # Resize back to original
#             pred_resized = cv2.resize(pred, (cv_image.shape[1], cv_image.shape[0]), interpolation=cv2.INTER_NEAREST)
#             self.publish_results(cv_image, pred_resized)
#         except Exception as e:
#             rospy.logerr(f"Image processing error: {str(e)}")

#     def publish_results(self, original_image, pred_mask):
#         colored_seg = self.color_map[pred_mask]
#         mono_mask = (pred_mask * 85).astype(np.uint8)

#         self.colored_seg_pub.publish(self.bridge.cv2_to_imgmsg(colored_seg, "bgr8"))
#         self.segmentation_pub.publish(self.bridge.cv2_to_imgmsg(mono_mask, "mono8"))
#         self.costmap_pub.publish(self.create_costmap(pred_mask))

#     def create_costmap(self, segmentation_mask):
#         costmap = OccupancyGrid()
#         costmap.header.stamp = rospy.Time.now()
#         costmap.header.frame_id = 'map'

#         size_x = int(20 / self.costmap_resolution)
#         size_y = int(20 / self.costmap_resolution)
#         costmap.info.resolution = self.costmap_resolution
#         costmap.info.width = size_x
#         costmap.info.height = size_y
#         costmap.info.origin = Pose(Point(-10.0, -17.0, 0), Quaternion(0, 0, 0, 1))

#         resized = cv2.resize(segmentation_mask, (size_x, size_y), interpolation=cv2.INTER_NEAREST)
#         costmap_data = np.full_like(resized, -1, dtype=np.int8)
#         costmap_data[resized == 0] = 0
#         costmap_data[resized == 1] = 50
#         costmap_data[resized == 2] = 100

#         obstacle_mask = (costmap_data == 100)
#         if np.any(obstacle_mask):
#             inflated = cv2.dilate(obstacle_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
#             costmap_data[(inflated == 1) & ~obstacle_mask] = 75

#         costmap.data = costmap_data.ravel().tolist()
#         return costmap


# if __name__ == '__main__':
#     try:
#         node = AIVisionNode()
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         pass


#-------------------------------------------------------------------------------------

# #Perfect
# #!/usr/bin/env python3
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# import rclpy
# from rclpy.node import Node
# import numpy as np
# import cv2
# import torch
# import torchvision
# from torchvision import transforms
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# from nav_msgs.msg import OccupancyGrid  
# from geometry_msgs.msg import Pose, Point, Quaternion 
# from std_msgs.msg import Header
# # import onnxruntime as ort 
# from autobot_ai.algorithms.AI_vision import FastSCNN

# class AIVisionNode(Node):
#     def __init__(self):
#         """Initialize the AI Vision Node for semantic segmentation and costmap generation"""
#         super().__init__('amr_ai_vision_node')
#         self.get_logger().info('Vision AI node initialization started.')
#         self.bridge = CvBridge()  # For ROS Image <-> OpenCV conversion

#         # Camera parameters (update with your actual values)
#         self.camera_height = 0.5  # meters
#         self.fx = 609.70
#         self.fy = 608.57
#         self.cx = 321.83
#         self.cy = 239.48
#         self.img_width = 640
#         self.img_height = 480
        
#         # BEV parameters
#         self.bev_width = 4.0    # meters
#         self.bev_height = 4.0   # meters
#         self.bev_resolution = 0.05  # meters/pixel
#         self.bev_px_width = int(self.bev_width / self.bev_resolution)
#         self.bev_px_height = int(self.bev_height / self.bev_resolution)
        
#         # Initialize homography matrix
#         self.H = self.calculate_homography()

#         # Initialize ONNX Runtime session for inference
#         try:
#             self.model = FastSCNN()  # Replace with your Fast-SCNN class
#             self.model.load_state_dict(torch.load("/home/jetson/autobot_ai_ws/src/autobot_ai/models/best_model_class4.pth", 
#                                                  map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
#                                                  weights_only=True))
#             self.model.eval()
#             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#             self.model.to(self.device)
#             self.get_logger().info("PyTorch model loaded successfully")

#         except Exception as e:
#             self.get_logger().error(f"Failed to load model weights .pth: {str(e)}")
#             raise

#         # Create subscribers and publishers
#         self.subscription = self.create_subscription(
#             Image,
#             '/camera/image_raw',
#             self.image_callback,
#             10
#         )
        
#         # Publisher for segmentation results (visualization)
#         self.segmentation_publisher = self.create_publisher(
#             Image, 
#             'segmentation_output',  
#             10
#         )
        
#         # Publisher for navigation costmap
#         self.costmap_publisher = self.create_publisher(
#             OccupancyGrid,
#             'semantic_costmap',
#             10
#         )

#         self.colored_segmentation_publisher = self.create_publisher(
#             Image, 
#             'colored_segmentation',  
#             10
#         )
        
#         self.birdseye_publisher = self.create_publisher(
#             Image,
#             'birdseye_view',
#             10
#         )

#     def calculate_homography(self):
#         """Calculate homography matrix for bird's-eye view transformation"""
#         # Define ground points in camera coordinate system (Z positive forward)
#         ground_pts = np.array([
#             [2.0, -2.0, self.camera_height],  # bottom-right
#             [2.0,  2.0, self.camera_height],  # top-right
#             [0.0,  2.0, self.camera_height],  # top-left
#             [0.0, -2.0, self.camera_height]   # bottom-left
#         ], dtype=np.float32)

#         # Project points to image plane
#         img_pts = []
#         for pt in ground_pts:
#             x, y, z = pt
#             u = (self.fx * x / z) + self.cx
#             v = (self.fy * y / z) + self.cy
#             img_pts.append([u, v])
#         img_pts = np.array(img_pts, dtype=np.float32)
        
#         # Define BEV destination points
#         bev_pts = np.array([
#             [self.bev_px_width-1, self.bev_px_height-1],  # bottom-right
#             [self.bev_px_width-1, 0],                     # top-right
#             [0, 0],                                        # top-left
#             [0, self.bev_px_height-1]                      # bottom-left
#         ], dtype=np.float32)
        
#         # Calculate homography
#         H, status = cv2.findHomography(img_pts, bev_pts)
#         if H is None:
#             self.get_logger().error("Homography calculation failed! Using identity")
#             return np.eye(3)
#         return H

#     def class_id_to_color(self, class_id):
#         """Map class IDs to BGR colors for visualization"""
#         color_map = {
#             0: (128, 64, 128),    # road (purple)
#             1: (244, 35, 232),    # sidewalk (Dark pink)
#             2: (107, 142, 35),    # vegetation (green)
#             3: (70, 130, 180),    # sky (sky blue)
#         }
#         return color_map.get(class_id, (0, 0, 0))  # Default to black for unknown classes

#     def create_colored_segmentation(self, pred_mask):
#         """Convert class IDs to colored image for visualization"""
#         # Create empty color image
#         height, width = pred_mask.shape
#         colored = np.zeros((height, width, 3), dtype=np.uint8)
        
#         # Apply colors based on class IDs
#         for class_id in np.unique(pred_mask):
#             mask = pred_mask == class_id
#             colored[mask] = self.class_id_to_color(class_id)
        
#         return colored

#     def image_callback(self, msg):
#         """Process incoming image messages and publish results"""
#         try:
#             # Convert ROS Image message to OpenCV format
#             cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
#             # Process image through neural network
#             pred_mask = self.process_image(cv_image)
            
#             # Publish colored segmentation
#             colored_seg = self.create_colored_segmentation(pred_mask)
#             colored_seg_msg = self.bridge.cv2_to_imgmsg(colored_seg, encoding="bgr8")
#             self.colored_segmentation_publisher.publish(colored_seg_msg)
            
#             # Publish mono segmentation (scaled for visualization)
#             vis_mask = (pred_mask * (255//3)).astype(np.uint8)  # Scale classes to 0-255
#             seg_msg = self.bridge.cv2_to_imgmsg(vis_mask, encoding="mono8")
#             self.segmentation_publisher.publish(seg_msg)
            
#             # Transform to bird's-eye view
#             bev_mask = cv2.warpPerspective(
#                 pred_mask.astype(np.float32), 
#                 self.H,
#                 (self.bev_px_width, self.bev_px_height),
#                 flags=cv2.INTER_NEAREST,
#                 borderValue=0  # Default to unknown
#             )
            
#             # Publish BEV visualization
#             bev_colored = self.create_colored_segmentation(bev_mask.astype(np.uint8))
#             bev_msg = self.bridge.cv2_to_imgmsg(bev_colored, encoding="bgr8")
#             bev_msg.header = msg.header
#             self.birdseye_publisher.publish(bev_msg)
            
#             # Convert segmentation to navigation costmap
#             costmap = self.create_costmap(bev_mask.astype(np.uint8))
#             self.costmap_publisher.publish(costmap)
            
#         except Exception as e:
#             self.get_logger().error(f"Error processing image: {str(e)}")

#     def process_image(self, image):
#         """Process image through neural network to get segmentation mask"""
#         # Define image preprocessing pipeline
#         transform = transforms.Compose([
#             transforms.ToPILImage(),  # Convert OpenCV image to PIL
#             transforms.Resize((512, 1024)),  # Resize to model's expected input
#             transforms.ToTensor(),  # Convert to tensor
#             transforms.Normalize(  # Normalize with ImageNet stats
#                 mean=[0.485, 0.456, 0.406], 
#                 std=[0.229, 0.224, 0.225]
#             )
#         ])

#         # Convert OpenCV image (BGR) to RGB
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # Preprocess image and convert to tensor
#         image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
#         image_tensor = image_tensor.to(self.device)

#         # Run inference
#         with torch.no_grad():
#             main_out, _ = self.model(image_tensor)
#             pred_classes = torch.argmax(torch.softmax(main_out, dim=1), dim=1).squeeze().cpu().numpy()

#         return pred_classes

#     def create_costmap(self, segmentation_mask):
#         """
#         Convert segmentation mask to navigation costmap
        
#         Args:
#             segmentation_mask: 2D numpy array with class predictions (BEV space)
            
#         Returns:
#             OccupancyGrid message ready for publishing
#         """
#         # Create OccupancyGrid message
#         costmap = OccupancyGrid()
        
#         # Set header information
#         costmap.header = Header()
#         costmap.header.stamp = self.get_clock().now().to_msg()
#         costmap.header.frame_id = "base_link"  # Relative to robot base
        
#         # Set costmap metadata
#         costmap.info.resolution = self.bev_resolution
#         costmap.info.width = self.bev_px_width
#         costmap.info.height = self.bev_px_height
        
#         # Set costmap origin (robot at center bottom)
#         costmap.info.origin = Pose(
#             position=Point(
#                 x=0.0,
#                 y=-self.bev_height/2,  # Center in Y direction
#                 z=0.0
#             ),
#             orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
#         )
        
#         # Initialize costmap data
#         costmap_data = np.full(
#             (self.bev_px_height, self.bev_px_width), 
#             100,  # Default to unknown/obstacle
#             dtype=np.int8
#         )
        
#         # Set free space where road is detected (class 0)
#         costmap_data[segmentation_mask == 0] = 0     # Road/free space
#         costmap_data[segmentation_mask == 1] = 50    # Sidewalk (medium cost)
#         costmap_data[segmentation_mask == 2] = 100   # Vegetation (obstacle)
#         costmap_data[segmentation_mask == 3] = -1    # Sky/unknown (ignored)
        
#         # Inflate obstacles for safety
#         kernel = np.ones((3, 3), np.uint8)
#         obstacles = (costmap_data == 100).astype(np.uint8)
#         inflated = cv2.dilate(obstacles, kernel, iterations=2)
        
#         # Apply inflation
#         costmap_data[inflated > 0] = 100
        
#         # Flatten and convert to list (required by OccupancyGrid)
#         costmap.data = costmap_data.flatten().tolist()
        
#         return costmap


# def main(args=None):
#     """Main function to initialize and run the node"""
#     rclpy.init(args=args)
#     node = AIVisionNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()




