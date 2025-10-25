#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

class EKFToggle:
    def __init__(self):
        rospy.init_node('ekf_toggle')
        self.gps_odom = rospy.Subscriber('/odometry/gps', Odometry, self.gps_cb)
        self.local_sub = rospy.Subscriber('/odometry/filtered/local', Odometry, self.local_cb)
        self.global_sub = rospy.Subscriber('/odometry/filtered/global', Odometry, self.global_cb)
        self.pub = rospy.Publisher('/odometry/filtered', Odometry, queue_size=1)
        self.last_gps = rospy.Time(0)
        self.gps_timeout = rospy.Duration(rospy.get_param('~gps_timeout', 2.0))

    def gps_cb(self, msg):
        self.last_gps = msg.header.stamp

    def local_cb(self, msg):
        self.try_publish(msg)

    def global_cb(self, msg):
        self.try_publish(msg)

    def try_publish(self, msg):
        now = rospy.Time.now()
        if now - self.last_gps < self.gps_timeout:
            if msg.header.frame_id == 'map':
                rospy.loginfo_throttle(1.0, "[EKF TOGGLE] Using GLOBAL EKF (GPS + IMU)")
                self.pub.publish(msg)
        else:
            if msg.header.frame_id == 'odom':
                rospy.logwarn_throttle(1.0, "[EKF TOGGLE] GPS lost or stale. Falling back to LOCAL EKF (VO + IMU)")
                self.pub.publish(msg)

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    EKFToggle().spin()

    
# #!/usr/bin/env python3
# import rospy
# import numpy as np
# from tf.transformations import quaternion_slerp
# from nav_msgs.msg import Odometry
# from geometry_msgs.msg import PoseWithCovariance, TwistWithCovariance

# class EKFToggle:
#     def __init__(self):
#         rospy.init_node('ekf_toggle')
        
#         # Load parameters
#         self.gps_timeout = rospy.Duration(rospy.get_param('~gps_timeout', 2.0))
#         self.max_data_age = rospy.Duration(rospy.get_param('~max_data_age', 0.5))
        
#         # Setup subscribers
#         self.gps_sub = rospy.Subscriber('/odometry/gps', Odometry, self.gps_cb, queue_size=10)
#         self.local_sub = rospy.Subscriber('/odometry/filtered/local', Odometry, self.local_cb, queue_size=10)
#         self.global_sub = rospy.Subscriber('/odometry/filtered/global', Odometry, self.global_cb, queue_size=10)
        
#         # Publisher
#         self.pub = rospy.Publisher('/odometry/filtered', Odometry, queue_size=10)
        
#         # State variables
#         self.last_gps = rospy.Time(0)
#         self.last_global = None
#         self.last_local = None
#         self.sequence = 0

#     def gps_cb(self, msg):
#         """Update last GPS timestamp"""
#         self.last_gps = msg.header.stamp

#     def global_cb(self, msg):
#         """Store global EKF output"""
#         self.last_global = msg
#         self.try_publish()

#     def local_cb(self, msg):
#         """Store local EKF output"""
#         self.last_local = msg
#         self.try_publish()

#     def interpolate_pose(self, pose1, pose2, alpha):
#         """SLERP interpolation with covariance blending"""
#         result = PoseWithCovariance()
        
#         # Linear position interpolation
#         result.pose.position.x = (1-alpha)*pose1.pose.position.x + alpha*pose2.pose.position.x
#         result.pose.position.y = (1-alpha)*pose1.pose.position.y + alpha*pose2.pose.position.y
#         result.pose.position.z = (1-alpha)*pose1.pose.position.z + alpha*pose2.pose.position.z
        
#         # SLERP for orientation
#         q1 = np.array([pose1.pose.orientation.x, pose1.pose.orientation.y,
#                       pose1.pose.orientation.z, pose1.pose.orientation.w])
#         q2 = np.array([pose2.pose.orientation.x, pose2.pose.orientation.y,
#                       pose2.pose.orientation.z, pose2.pose.orientation.w])
#         q_result = quaternion_slerp(q1, q2, alpha)
#         result.pose.orientation.x = q_result[0]
#         result.pose.orientation.y = q_result[1]
#         result.pose.orientation.z = q_result[2]
#         result.pose.orientation.w = q_result[3]
        
#         # Covariance blending
#         result.covariance = [
#             (1-alpha)*pose1.covariance[i] + alpha*pose2.covariance[i] 
#             for i in range(36)
#         ]
        
#         return result

#     def try_publish(self):
#         """Publish blended odometry based on GPS status"""
#         if None in (self.last_global, self.last_local):
#             return
            
#         now = rospy.Time.now()
        
#         # Check data freshness
#         if (now - self.last_global.header.stamp) > self.max_data_age:
#             rospy.logwarn_throttle(1.0, "Stale global EKF data")
#             return
#         if (now - self.last_local.header.stamp) > self.max_data_age:
#             rospy.logwarn_throttle(1.0, "Stale local EKF data")
#             return
            
#         # Calculate blend factor
#         time_since_gps = now - self.last_gps
#         blend_factor = min(1.0, time_since_gps.to_sec() / self.gps_timeout.to_sec())
        
#         # Prepare output message
#         output = Odometry()
#         output.header.stamp = now
#         output.header.seq = self.sequence
#         self.sequence += 1
#         output.header.frame_id = 'map' if blend_factor < 0.5 else 'odom'
#         output.child_frame_id = 'base_link'
        
#         # Blend data
#         output.pose = self.interpolate_pose(self.last_global.pose, self.last_local.pose, blend_factor)
#         output.twist = self.last_global.twist if blend_factor < 0.5 else self.last_local.twist
        
#         # Status logging
#         if blend_factor < 0.2:
#             rospy.loginfo_throttle(1.0, "[EKF] Using GLOBAL EKF (GPS available)")
#         elif blend_factor > 0.8:
#             rospy.logwarn_throttle(1.0, "[EKF] Using LOCAL EKF (GPS stale/lost)")
#         else:
#             rospy.loginfo_throttle(1.0, f"[EKF] Blending ({int(100*(1-blend_factor))}% global)")
        
#         self.pub.publish(output)

# if __name__ == '__main__':
#     try:
#         EKFToggle().spin()
#     except rospy.ROSInterruptException:
#         pass