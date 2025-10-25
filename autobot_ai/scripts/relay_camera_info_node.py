#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from your_package_name.msg import StereoCameraInfo  # Replace with your package name

class CameraInfoRelay(Node):
    def __init__(self):
        super().__init__('camera_info_relay')

        # Subscribers
        self.left_sub = self.create_subscription(
            CameraInfo, '/rleft/camera_info', self.left_callback, 10)
        self.right_sub = self.create_subscription(
            CameraInfo, '/rright/camera_info', self.right_callback, 10)

        # Publisher for combined info
        self.stereo_pub = self.create_publisher(StereoCameraInfo, '/stereo/camera_info', 10)

        self.left_info = None
        self.right_info = None

    def left_callback(self, msg):
        self.left_info = msg
        self.publish_combined()

    def right_callback(self, msg):
        self.right_info = msg
        self.publish_combined()

    def publish_combined(self):
        if self.left_info and self.right_info:
            msg = StereoCameraInfo()
            msg.left = self.left_info
            msg.right = self.right_info
            self.stereo_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = CameraInfoRelay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()