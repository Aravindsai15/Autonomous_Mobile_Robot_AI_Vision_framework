#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

class OdomToPose:
    def __init__(self):
        rospy.init_node('odom_to_pose_node', anonymous=True)
        
        self.sub = rospy.Subscriber(
            '/localization/filtered_odometry', 
            Odometry, 
            self.callback,
            queue_size=10)
            
        self.pub = rospy.Publisher(
            '/localization/filtered_odom',
            PoseStamped,
            queue_size=10)

    def callback(self, msg):
        pose_msg = PoseStamped()
        pose_msg.header = msg.header
        pose_msg.pose = msg.pose.pose
        self.pub.publish(pose_msg)

if __name__ == '__main__':
    try:
        converter = OdomToPose()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass