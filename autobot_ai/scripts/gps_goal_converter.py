#!/usr/bin/env python3
import rospy
import utm
import os
import yaml
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import NavSatFix

class GPSGoalConverter:
    def __init__(self):
        rospy.init_node('gps_goal_converter')
        
        # Parameters
        self.world_frame = rospy.get_param('~world_frame', 'odom')
        self.origin_file = os.path.expanduser(rospy.get_param('~origin_file', '~/.ros/gps_origin.yaml'))
        
        # Origin management
        self.origin_utm = None
        self.origin_set = False
        self.load_origin()
        
        # ROS interfaces
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        rospy.Subscriber('/gps_goal_in', NavSatFix, self.gps_callback)
        
        rospy.loginfo("GPS Goal Converter ready")

    def load_origin(self):
        """Load origin from YAML file if exists"""
        try:
            if os.path.exists(self.origin_file):
                with open(self.origin_file, 'r') as f:
                    data = yaml.safe_load(f)
                    self.origin_utm = (data['easting'], data['northing'], data['zone_num'], data['zone_letter'])
                    self.origin_set = True
                    rospy.loginfo(f"Loaded origin from file: {self.origin_utm[:2]}")
        except Exception as e:
            rospy.logwarn(f"Failed to load origin: {str(e)}")
            self.origin_set = False

    def save_origin(self, utm_coords):
        """Save origin to YAML file"""
        try:
            os.makedirs(os.path.dirname(self.origin_file), exist_ok=True)
            with open(self.origin_file, 'w') as f:
                yaml.dump({
                    'easting': utm_coords[0],
                    'northing': utm_coords[1],
                    'zone_num': utm_coords[2],
                    'zone_letter': utm_coords[3]
                }, f)
        except Exception as e:
            rospy.logerr(f"Failed to save origin: {str(e)}")

    def gps_callback(self, msg):
        """Handle incoming GPS messages"""
        if msg.status.status < 0:
            rospy.logwarn("Received invalid GPS fix")
            return
            
        try:
            # Convert to UTM
            current_utm = utm.from_latlon(msg.latitude, msg.longitude)
            
            # Set origin if not set
            if not self.origin_set:
                self.origin_utm = current_utm
                self.save_origin(current_utm)
                self.origin_set = True
                rospy.loginfo(f"New origin set at UTM: {current_utm[0]:.3f}, {current_utm[1]:.3f}")
                return
                
            # Calculate relative position
            rel_x = current_utm[0] - self.origin_utm[0]
            rel_y = current_utm[1] - self.origin_utm[1]
            
            # Create goal message
            goal = PoseStamped()
            goal.header.stamp = rospy.Time.now()
            goal.header.frame_id = self.world_frame
            goal.pose.position.x = rel_x
            goal.pose.position.y = rel_y
            goal.pose.orientation.w = 1.0  # Neutral orientation
            
            rospy.loginfo(f"Publishing goal at: X: {rel_x:.2f}m, Y: {rel_y:.2f}m")
            self.goal_pub.publish(goal)
            
        except Exception as e:
            rospy.logerr(f"Error processing GPS data: {str(e)}")

if __name__ == '__main__':
    try:
        GPSGoalConverter()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("GPS Goal Converter shutdown")