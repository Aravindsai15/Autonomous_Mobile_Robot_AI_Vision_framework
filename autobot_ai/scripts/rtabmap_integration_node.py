#!/usr/bin/env python3
# #!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node
# from nav_msgs.msg import OccupancyGrid
# from rtabmap_msgs.msg import UserData
# from geometry_msgs.msg import PoseStamped
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import numpy as np

# class RTABMapIntegration(Node):
#     def __init__(self):
#         super().__init__('rtabmap_integration')
        
#         # Subscribers
#         self.create_subscription(
#             OccupancyGrid,
#             '/semantic_costmap',
#             self.semantic_callback,
#             10)
            
#         self.create_subscription(
#             Image,
#             '/camera/image_raw',
#             self.image_callback,
#             10)
            
#         self.create_subscription(
#             PoseStamped,
#             '/rtabmap/localization_pose',  # RTAB-Map's pose output
#             self.pose_callback,
#             10)
        
#         # Publishers
#         self.user_data_pub = self.create_publisher(
#             UserData,
#             '/rtabmap/user_data',
#             10)
            
#         self.fused_costmap_pub = self.create_publisher(
#             OccupancyGrid,
#             '/fused_costmap',
#             10)
            
#         self.robot_pose_pub = self.create_publisher(
#             PoseStamped,
#             '/robot_pose',
#             10)
            
#         self.bridge = CvBridge()
#         self.latest_semantic = None
#         self.rtabmap_pose = None
#         self.rtabmap_confidence = 0.0

#     def semantic_callback(self, msg):
#         """Process semantic costmap and convert to RTAB-Map format"""
#         self.latest_semantic = msg
        
#         try:
#             # Convert to RTAB-Map UserData
#             user_data = UserData()
#             user_data.header = msg.header
#             user_data.rows = msg.info.height
#             user_data.cols = msg.info.width
#             user_data.type = 5  # CV_8UC1
            
#             # Convert costmap data to bytes
#             costmap_array = np.array(msg.data, dtype=np.uint8)
#             user_data.data = costmap_array.tobytes()
            
#             self.user_data_pub.publish(user_data)
            
#             # Publish fused costmap if we have RTAB-Map data
#             if self.rtabmap_pose:
#                 self.publish_fused_costmap()
                
#         except Exception as e:
#             self.get_logger().error(f"Semantic processing failed: {str(e)}")

#     def pose_callback(self, msg):
#         """Handle RTAB-Map pose updates"""
#         self.rtabmap_pose = msg
#         self.rtabmap_confidence = msg.pose.covariance[0]  # Example confidence metric
        
#         # Publish standardized robot pose
#         robot_pose = PoseStamped()
#         robot_pose.header = msg.header
#         robot_pose.pose = msg.pose
#         self.robot_pose_pub.publish(robot_pose)
        
#         # Update costmap when new pose arrives
#         if self.latest_semantic:
#             self.publish_fused_costmap()

#     def image_callback(self, msg):
#         """Pass through camera images with timestamp alignment"""
#         # Can add image preprocessing here if needed
#         pass

#     def publish_fused_costmap(self):
#         """Fuse semantic and RTAB-Map costmaps"""
#         try:
#             if not self.latest_semantic:
#                 return
                
#             # Create basic fused costmap (semantic takes priority)
#             semantic_data = np.array(self.latest_semantic.data).reshape(
#                 self.latest_semantic.info.height,
#                 self.latest_semantic.info.width)
                
#             # For now just use semantic, add RTAB-Map fusion logic later
#             fused_data = semantic_data  # Placeholder
            
#             # Create OccupancyGrid message
#             fused_msg = OccupancyGrid()
#             fused_msg.header = self.latest_semantic.header
#             fused_msg.header.stamp = self.get_clock().now().to_msg()
#             fused_msg.info = self.latest_semantic.info
#             fused_msg.data = fused_data.flatten().tolist()
            
#             self.fused_costmap_pub.publish(fused_msg)
            
#         except Exception as e:
#             self.get_logger().error(f"Costmap fusion failed: {str(e)}")

# def main(args=None):
#     rclpy.init(args=args)
#     node = RTABMapIntegration()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()




#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from nav_msgs.msg import OccupancyGrid
from rtabmap_msgs.msg import UserData
import numpy as np

class RTABMapIntegrationNode(Node):
    def __init__(self):
        super().__init__('rtabmap_integration_node')

        # Declare parameters with correct descriptors
        self.declare_parameter(
            'input_topic',
            '/ai/semantic_costmap',
            ParameterDescriptor(description='Input costmap topic')
        )
        self.declare_parameter(
            'output_topic',
            '/rtabmap/user_data',
            ParameterDescriptor(description='Output user data topic')
        )
        self.declare_parameter(
            'queue_size',
            10,
            ParameterDescriptor(description='Subscription queue size')
        )
        self.declare_parameter(
            'unknown_value',
            100,
            ParameterDescriptor(description='Value for unknown cells (-1)')
        )
        self.declare_parameter(
            'debug',
            False,
            ParameterDescriptor(description='Enable debug output')
        )

        # Get parameters
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        queue_size = self.get_parameter('queue_size').get_parameter_value().integer_value
        self.unknown_value = self.get_parameter('unknown_value').get_parameter_value().integer_value
        self.debug_enabled = self.get_parameter('debug').get_parameter_value().bool_value

        # Create publishers and subscribers
        self.subscription = self.create_subscription(
            OccupancyGrid,
            input_topic,
            self.costmap_callback,
            queue_size
        )

        self.publisher = self.create_publisher(
            UserData,
            output_topic,
            queue_size
        )

        # Conditional debug publisher
        self.debug_publisher = None
        if self.debug_enabled:
            self.debug_publisher = self.create_publisher(
                OccupancyGrid,
                '/debug/semantic_to_user_data',
                queue_size
            )

        self.get_logger().info(
            f"Node initialized:\n"
            f"  Input topic: {input_topic}\n"
            f"  Output topic: {output_topic}\n"
            f"  Queue size: {queue_size}"
        )
            
    def costmap_callback(self, msg):
        try:
            start_time = self.get_clock().now()

            if not msg.data:
                self.get_logger().warn("Received empty costmap data")
                return

            # Prepare UserData message
            user_data = UserData()
            user_data.header = msg.header
            user_data.rows = 1  # RTABMap expects flattened data
            user_data.cols = len(msg.data)
            user_data.type = 5  # CV_8SC1 (signed 8-bit)

            # Convert data to signed bytes (-1 to 100)
            converted_data = np.array([
                -1 if val == -1 else min(100, max(0, int(val)))
                for val in msg.data
            ], dtype=np.int8)
            
            user_data.data = bytes(converted_data)
            self.publisher.publish(user_data)

            # Debug output
            if self.debug_enabled and self.debug_publisher:
                debug_msg = OccupancyGrid()
                debug_msg.header = msg.header
                debug_msg.info = msg.info
                debug_msg.data = [int(x) for x in converted_data]  # Convert back to int list
                self.debug_publisher.publish(debug_msg)

            duration = (self.get_clock().now() - start_time).nanoseconds / 1e6
            self.get_logger().debug(f"Processed costmap in {duration:.2f} ms")

        except Exception as e:
            self.get_logger().error(f"Costmap processing error: {str(e)}")
            
def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = RTABMapIntegrationNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info("Node shutdown requested.")
    except Exception as e:
        if node:
            node.get_logger().error(f"Node crashed: {e}")
        else:
            print(f"Node crashed before initialization: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
