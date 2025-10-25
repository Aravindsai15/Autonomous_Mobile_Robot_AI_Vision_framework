#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
import random

class GPSSimulator(Node):
    def __init__(self):
        super().__init__('gps_simulator')
        self.publisher = self.create_publisher(NavSatFix, '/gps/fix', 10)
        
        # Initial coordinates (set your Google Maps location here)
        self.latitude = 47.6062  # Example: Seattle coordinates
        self.longitude = -122.3321
        self.gps_active = True
        
        # Simulate GPS availability
        self.create_timer(1.0, self.publish_gps)
    
    def publish_gps(self):
        msg = NavSatFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'gps'
        
        if self.gps_active:
            # Simulate small GPS drift
            msg.latitude = self.latitude + random.uniform(-0.0001, 0.0001)
            msg.longitude = self.longitude + random.uniform(-0.0001, 0.0001)
            msg.status.status = 1  # GPS fix available
        else:
            msg.status.status = 0  # No fix
            
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = GPSSimulator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()