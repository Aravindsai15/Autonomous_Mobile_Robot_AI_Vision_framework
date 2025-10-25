#!/usr/bin/env python2


#!/usr/bin/env python2
import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

class DynamicGoalAdjuster:
    def __init__(self):
        rospy.init_node('dynamic_goal_adjuster')
        
        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Current robot state (in map frame)
        self.current_pose_map = None
        self.initial_goal = None
        self.initial_pose_map = None
        
        # Subscribers
        self.odom_sub = rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback)
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        
        # Publisher for adjusted goals
        self.adjusted_goal_pub = rospy.Publisher('/adjusted_goal', PoseStamped, queue_size=1)
        
        rospy.loginfo("Dynamic Goal Adjuster initialized in map frame")

    def odom_callback(self, msg):
        """Convert odom pose to map frame and store"""
        try:
            # Create pose stamped in odom frame
            pose_odom = PoseStamped()
            pose_odom.header = msg.header
            pose_odom.pose = msg.pose.pose
            
            # Transform to map frame
            transform = self.tf_buffer.lookup_transform('map', 'odom', rospy.Time(0))
            pose_map = tf2_geometry_msgs.do_transform_pose(pose_odom, transform)
            
            self.current_pose_map = pose_map.pose
            if self.initial_goal is not None and self.initial_pose_map is None:
                self.initial_pose_map = self.current_pose_map
                
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("TF error: %s" % str(e))
        except Exception as e:
            rospy.logerr("Odom callback error: %s" % str(e))

    def goal_callback(self, msg):
        """Handle new goal from Rviz (already in map frame)"""
        try:
            if msg.header.frame_id != 'map':
                rospy.logwarn("Goal not in map frame! Got %s" % msg.header.frame_id)
                return
                
            self.initial_goal = msg
            self.initial_pose_map = None  # Reset to capture new initial pose
            rospy.loginfo("New goal received at (%.2f, %.2f)" % (msg.pose.position.x, msg.pose.position.y))
            
        except Exception as e:
            rospy.logerr("Goal callback error: %s" % str(e))

    def adjust_goal(self):
        """Calculate and publish adjusted goal in map frame"""
        try:
            if None in [self.initial_goal, self.initial_pose_map, self.current_pose_map]:
                return

            # Calculate movement in map frame
            dx = self.current_pose_map.position.x - self.initial_pose_map.position.x
            dy = self.current_pose_map.position.y - self.initial_pose_map.position.y
            
            # Adjust goal position
            adjusted_goal = PoseStamped()
            adjusted_goal.header.stamp = rospy.Time.now()
            adjusted_goal.header.frame_id = "map"
            adjusted_goal.pose.position.x = self.initial_goal.pose.position.x - dx
            adjusted_goal.pose.position.y = self.initial_goal.pose.position.y - dy
            adjusted_goal.pose.orientation = self.initial_goal.pose.orientation
            self.adjusted_goal_pub.publish(adjusted_goal)
            
        except Exception as e:
            rospy.logerr("Goal adjustment error: %s" % str(e))

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.adjust_goal()
            rate.sleep()

if __name__ == '__main__':
    node = DynamicGoalAdjuster()
    node.run()



# import rospy
# import tf2_ros
# import tf2_geometry_msgs
# from geometry_msgs.msg import PoseStamped
# from nav_msgs.msg import Odometry

# class DynamicGoalAdjuster:
#     def __init__(self):
#         rospy.init_node('dynamic_goal_adjuster')
        
#         # TF setup
#         self.tf_buffer = tf2_ros.Buffer()
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
#         # Current robot state (in map frame)
#         self.current_pose_map = None
#         self.initial_goal = None
#         self.initial_pose_map = None
        
#         # Subscribers
#         self.odom_sub = rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback)
#         self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        
#         # Publisher for adjusted goals
#         self.adjusted_goal_pub = rospy.Publisher('/adjusted_goal', PoseStamped, queue_size=1)
        
#         rospy.loginfo("Dynamic Goal Adjuster initialized in map frame")

#     def odom_callback(self, msg):
#         """Convert odom pose to map frame and store"""
#         try:
#             # Create pose stamped in odom frame
#             pose_odom = PoseStamped()
#             pose_odom.header = msg.header
#             pose_odom.pose = msg.pose.pose
            
#             # Transform to map frame
#             transform = self.tf_buffer.lookup_transform('map', 'odom', rospy.Time(0))
#             pose_map = tf2_geometry_msgs.do_transform_pose(pose_odom, transform)
            
#             self.current_pose_map = pose_map.pose
#             if self.initial_goal is not None and self.initial_pose_map is None:
#                 self.initial_pose_map = self.current_pose_map
                
#         except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
#                 tf2_ros.ExtrapolationException) as e:
#             rospy.logwarn(f"TF error: {str(e)}")
#         except Exception as e:
#             rospy.logerr(f"Odom callback error: {str(e)}")

#     def goal_callback(self, msg):
#         """Handle new goal from Rviz (already in map frame)"""
#         try:
#             if msg.header.frame_id != 'map':
#                 rospy.logwarn(f"Goal not in map frame! Got {msg.header.frame_id}")
#                 return
                
#             self.initial_goal = msg
#             self.initial_pose_map = None  # Reset to capture new initial pose
#             rospy.loginfo(f"New goal received at ({msg.pose.position.x}, {msg.pose.position.y})")
            
#         except Exception as e:
#             rospy.logerr(f"Goal callback error: {str(e)}")

#     def adjust_goal(self):
#         """Calculate and publish adjusted goal in map frame"""
#         try:
#             if None in [self.initial_goal, self.initial_pose_map, self.current_pose_map]:
#                 return

#             # Calculate movement in map frame
#             dx = self.current_pose_map.position.x - self.initial_pose_map.position.x
#             dy = self.current_pose_map.position.y - self.initial_pose_map.position.y
            
#             # Adjust goal position
#             adjusted_goal = PoseStamped()
#             adjusted_goal.header.stamp = rospy.Time.now()
#             adjusted_goal.header.frame_id = "map"
#             adjusted_goal.pose.position.x = self.initial_goal.pose.position.x - dx
#             adjusted_goal.pose.position.y = self.initial_goal.pose.position.y - dy
#             adjusted_goal.pose.orientation = self.initial_goal.pose.orientation
#             self.adjusted_goal_pub.publish(self.initial_goal)

            
#         except Exception as e:
#             rospy.logerr(f"Goal adjustment error: {str(e)}")

#     def run(self):
#         rate = rospy.Rate(10)
#         while not rospy.is_shutdown():
#             self.adjust_goal()
#             rate.sleep()

# if __name__ == '__main__':
#     node = DynamicGoalAdjuster()
#     node.run()