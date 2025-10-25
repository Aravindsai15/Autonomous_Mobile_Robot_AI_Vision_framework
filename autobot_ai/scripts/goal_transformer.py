#!/usr/bin/env python3
import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

class GoalTransformer:
    def __init__(self):
        rospy.init_node('goal_transformer')
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Subscribers
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback)
        
        # Publisher
        self.transformed_goal_pub = rospy.Publisher('/transformed_goal', PoseStamped, queue_size=10)
        
        self.current_pose = None
        rospy.loginfo("Goal Transformer ready")

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def goal_callback(self, msg):
        try:
            if self.current_pose is None:
                rospy.logwarn("No odometry received yet")
                return
            
            # Transform goal to map frame if needed
            if msg.header.frame_id != 'map':
                transform = self.tf_buffer.lookup_transform('map', 
                                                          msg.header.frame_id,
                                                          rospy.Time(0),
                                                          rospy.Duration(1.0))
                msg = tf2_geometry_msgs.do_transform_pose(msg, transform)
            
            # Adjust goal based on current semantic understanding
            adjusted_goal = PoseStamped()
            adjusted_goal.header.stamp = rospy.Time.now()
            adjusted_goal.header.frame_id = 'map'
            
            # Keep original goal orientation
            adjusted_goal.pose.orientation = msg.pose.orientation
            
            # Adjust position based on current robot pose
            if self.current_pose:
                dx = msg.pose.position.x - self.current_pose.position.x
                dy = msg.pose.position.y - self.current_pose.position.y
                adjusted_goal.pose.position.x = self.current_pose.position.x + dx
                adjusted_goal.pose.position.y = self.current_pose.position.y + dy
            else:
                adjusted_goal.pose.position = msg.pose.position
            
            self.transformed_goal_pub.publish(adjusted_goal)
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"TF error: {str(e)}")
        except Exception as e:
            rospy.logerr(f"Goal transformation error: {str(e)}")

if __name__ == '__main__':
    node = GoalTransformer()
    rospy.spin()