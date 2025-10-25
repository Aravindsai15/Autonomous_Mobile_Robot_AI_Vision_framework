#!/usr/bin/env python2

# -*- coding: utf-8 -*-

import rospy
import numpy as np
import math
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Twist, Pose
from std_msgs.msg import Header
from sensor_msgs.msg import NavSatFix
from navfn.srv import MakeNavPlan, MakeNavPlanRequest
from pyproj import Proj
import tf2_ros
from tf2_ros import transform_listener
from tf.transformations import euler_from_quaternion

class NavigationManager:
    def __init__(self):
        rospy.init_node('navigation_manager')
        
        # Parameters
        self.use_gps_goals = rospy.get_param('~use_gps_goals', True)
        self.costmap_topic = rospy.get_param('~costmap_topic', '/semantic_costmap')
        self.goal_tolerance = rospy.get_param('~goal_tolerance', 0.5)
        self.replan_interval = rospy.get_param('~replan_interval', 10.0)
        self.max_deviation = rospy.get_param('~max_deviation', 1.0)
        self.utm_zone = rospy.get_param('~utm_zone', '32N')
        
        # State variables
        self.costmap = None
        self.current_pose = None
        self.goal_utm = None
        self.global_path = []
        self.original_path = []
        self.current_wp_idx = 0
        self.last_plan_time = rospy.Time.now()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.utm_to_map_transform = None
        
        # UTM projector
        self.utm_proj = Proj(proj='utm', zone=self.utm_zone, ellps='WGS84')
        
        # Subscribers
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.rviz_goal_cb)
        rospy.Subscriber('/gps_goal', PoseStamped, self.gps_goal_cb)
        rospy.Subscriber(self.costmap_topic, OccupancyGrid, self.semantic_callback)
        rospy.Subscriber('/odometry/filtered', Odometry, self.odom_cb)
        rospy.Subscriber('/sensors/gps/fix', NavSatFix, self.gps_status_cb)
        
        # Service client
        rospy.wait_for_service('/global_planner/make_plan')
        self.plan_srv = rospy.ServiceProxy('/global_planner/make_plan', MakeNavPlan)
        
        # Publishers
        self.path_pub = rospy.Publisher('/global_plan', Path, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.status_pub = rospy.Publisher('/nav_status', Header, queue_size=1)
        
        # Timer
        rospy.Timer(rospy.Duration(0.1), self.control_loop)
        rospy.loginfo("NavigationManager ready (using semantic-aware A*)")
        
        # Initialize transform (assuming map and UTM are aligned at origin)
        self.utm_to_map_transform = (0, 0, 0)  # (dx, dy, dyaw)

    def rviz_goal_cb(self, msg):
        """Handle RViz clicked goals"""
        if self.use_gps_goals:
            rospy.logwarn("Ignoring RViz goal in GPS mode")
            return
            
        self.goal_utm = (msg.pose.position.x, msg.pose.position.y)
        rospy.loginfo("RViz goal set: %.2f, %.2f", *self.goal_utm)
        self.plan_path()

    def gps_goal_cb(self, msg):
        """Convert GPS goals to UTM and plan path"""
        lat, lon = msg.pose.position.x, msg.pose.position.y
        try:
            e, n = self.utm_proj(lon, lat)
            # Apply transform to map frame
            if self.utm_to_map_transform:
                dx, dy, _ = self.utm_to_map_transform
                e += dx
                n += dy
            self.goal_utm = (e, n)
            rospy.loginfo("GPS goal set: UTM(%.2f, %.2f)", e, n)
            self.plan_path()
        except Exception as e:
            rospy.logerr("GPS->UTM conversion failed: %s", e)

    def gps_status_cb(self, msg):
        """Monitor GPS status for recovery behaviors"""
        # Implement GPS health monitoring here
        pass

    def odom_cb(self, msg):
        """Update current robot pose"""
        self.current_pose = msg.pose.pose

    def semantic_callback(self, msg):
        """Process semantic costmap with proper value scaling"""
        self.costmap = msg
        

    def validate_path(self, path):
        """Ensure path stays on drivable areas using existing semantic costmap"""
        if not self.costmap:
            rospy.logwarn("No costmap for validation")
            return False
            
        origin = self.costmap.info.origin.position
        res = self.costmap.info.resolution
        width = self.costmap.info.width
        height = self.costmap.info.height
        
        for x, y in path:
            # Convert to grid coordinates
            mx = int((x - origin.x) / res)
            my = int((y - origin.y) / res)
            
            # Check bounds
            if not (0 <= mx < width and 0 <= my < height):
                rospy.logwarn("Path point out of costmap bounds")
                return False
                
            # Check cost value (using pre-processed semantic map)
            idx = my * width + mx
            
            # CRITICAL FIX: Handle unknown (-1) and obstacles (>=50)
            cost_value = self.costmap.data[idx]
            if cost_value == -1 or cost_value >= 50:  # Non-drivable areas
                rospy.logwarn("Path intersects non-drivable area at (%d, %d)", mx, my)
                return False
                
        return True

    def plan_path(self):
        """Generate and validate path using global planner"""
        # Check system readiness
        if not self.wait_for_costmap(timeout=5.0):
            rospy.logerr("Costmap not available for planning")
            return False
            
        if not self.current_pose:
            rospy.logwarn("No current pose for planning")
            return False
            
        if not self.goal_utm:
            rospy.logwarn("No goal set for planning")
            return False
            
        # Prepare planning request
        req = MakeNavPlanRequest()
        req.start = PoseStamped(
            header=Header(stamp=rospy.Time.now(), frame_id="map"),
            pose=self.current_pose
        )
        
        goal_pose = Pose()
        goal_pose.position.x = self.goal_utm[0]
        goal_pose.position.y = self.goal_utm[1]
        goal_pose.orientation.w = 1.0  # Neutral orientation
        
        req.goal = PoseStamped(
            header=Header(stamp=rospy.Time.now(), frame_id="map"),
            pose=goal_pose
        )
        
        try:
            # Call planner service
            resp = self.plan_srv(req)
            
            if not resp.plan.poses:
                rospy.logwarn("Planner returned empty path")
                return False
                
            # Extract path waypoints
            path_points = [(p.pose.position.x, p.pose.position.y) 
                          for p in resp.plan.poses]
            
            # Validate path against semantic costmap
            if not self.validate_path(path_points):
                rospy.logwarn("Path validation failed - replanning")
                return self.recover_and_replan()
                
            # Store valid path
            self.original_path = path_points
            self.global_path = list(path_points)
            self.current_wp_idx = 0
            self.last_plan_time = rospy.Time.now()
            
            # Publish for visualization
            self.publish_path(self.global_path)
            rospy.loginfo("Valid path planned with %d waypoints", len(self.global_path))
            return True
            
        except rospy.ServiceException as e:
            rospy.logerr("Plan service failed: %s", e)
            return False

    def recover_and_replan(self):
        """Execute recovery behavior and attempt replan"""
        rospy.loginfo("Executing recovery behavior")
        
        # 1. Stop the robot
        self.stop_robot()
        
        # 2. Small backward movement
        cmd = Twist()
        cmd.linear.x = -0.1
        self.cmd_vel_pub.publish(cmd)
        rospy.sleep(0.5)
        
        # 3. Rotate in place
        cmd = Twist()
        cmd.angular.z = 0.3
        self.cmd_vel_pub.publish(cmd)
        rospy.sleep(1.0)
        
        # 4. Reattempt planning
        return self.plan_path()

    def publish_path(self, path):
        """Publish path for visualization in RViz"""
        msg = Path()
        msg.header = Header(stamp=rospy.Time.now(), frame_id="map")
        
        for x, y in path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position = Point(x, y, 0)
            ps.pose.orientation = Quaternion(0, 0, 0, 1)
            msg.poses.append(ps)
            
        self.path_pub.publish(msg)

    def stop_robot(self):
        """Halt all robot movement"""
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)

    def get_yaw(self, orientation):
        """Extract yaw from quaternion"""
        _, _, yaw = euler_from_quaternion([
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        ])
        return yaw

    def path_deviation(self):
        """Calculate max deviation from planned path"""
        if not self.current_pose or not self.global_path:
            return 0
            
        current_pos = (self.current_pose.position.x, 
                      self.current_pose.position.y)
        min_dist = float('inf')
        
        for point in self.global_path:
            dist = math.hypot(point[0]-current_pos[0], point[1]-current_pos[1])
            if dist < min_dist:
                min_dist = dist
                
        return min_dist

    def wait_for_costmap(self, timeout=5.0):
        """Wait for valid costmap to become available"""
        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time).to_sec() < timeout:
            if self.costmap and self.costmap.info.resolution > 0:
                return True
            rospy.sleep(0.1)
        return False

    def control_loop(self, event):
        """Main navigation control loop"""
        # Periodically replan if needed
        if (rospy.Time.now() - self.last_plan_time).to_sec() > self.replan_interval:
            rospy.loginfo("Periodic replanning triggered")
            self.plan_path()
            
        # Check if we have a valid path to follow
        if not self.global_path or self.current_wp_idx >= len(self.global_path):
            return
            
        # Check for significant deviation
        if self.path_deviation() > self.max_deviation:
            rospy.logwarn("Path deviation too large, replanning")
            self.plan_path()
            return
            
        # Get current target waypoint
        wx, wy = self.global_path[self.current_wp_idx]
        
        # Get current position and orientation
        if not self.current_pose:
            return
            
        cx = self.current_pose.position.x
        cy = self.current_pose.position.y
        yaw = self.get_yaw(self.current_pose.orientation)
        
        # Calculate distance and angle to waypoint
        distance = math.hypot(wx - cx, wy - cy)
        target_angle = math.atan2(wy - cy, wx - cx)
        angle_error = (target_angle - yaw + math.pi) % (2 * math.pi) - math.pi
        
        # Move to next waypoint if close enough
        if distance < self.goal_tolerance:
            self.current_wp_idx += 1
            if self.current_wp_idx >= len(self.global_path):
                rospy.loginfo("Final waypoint reached!")
                self.stop_robot()
                self.publish_status("GOAL_REACHED")
            return
            
        # Generate control command
        cmd = Twist()
        
        # Adaptive speed based on alignment
        if abs(angle_error) > math.pi/4:  # >45° misalignment
            cmd.linear.x = 0.1
        else:
            cmd.linear.x = min(0.5, distance * 0.5)
            
        # Angular control (proportional)
        cmd.angular.z = 1.0 * angle_error
        
        # Publish command
        self.cmd_vel_pub.publish(cmd)

    def publish_status(self, status):
        """Publish navigation status"""
        msg = Header()
        msg.stamp = rospy.Time.now()
        msg.frame_id = status
        self.status_pub.publish(msg)

if __name__ == '__main__':
    try:
        NavigationManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


        # def plan_path(self):
        # # 1) sanity checks
        # if not (self.costmap and self.current_pose and self.goal_utm):
        #     rospy.logwarn("Missing costmap, pose, or goal → skipping")
        #     return

        # # 2) turn grid into [H,W] array
        # grid = np.array(self.costmap.data, dtype=np.int8) \
        #         .reshape(self.costmap.info.height,
        #                     self.costmap.info.width)

        # # 3) start/goal in grid indices
        # def world_to_map(x,y):
        #     mx = int((x - self.costmap.info.origin.position.x) / self.costmap.info.resolution)
        #     my = int((y - self.costmap.info.origin.position.y) / self.costmap.info.resolution)
        #     return np.clip(mx, 0, grid.shape[1]-1), np.clip(my, 0, grid.shape[0]-1)

        # sx, sy = self.current_pose.position.x, self.current_pose.position.y
        # gx, gy = self.goal_utm
        # start = world_to_map(sx, sy)
        # goal  = world_to_map(gx, gy)

        # # 4) cost function: roads=1, sidewalk=10, obstacles blocked
        # def cell_cost(u,v):
        #     val = grid[v,u]
        #     if val >= 100:    # obstacle
        #         return None
        #     elif val ==   0:  # road
        #         return 1
        #     elif val ==  50:  # sidewalk
        #         return 10
        #     else:             # unknown
        #         return 20

        # # 5) A* boilerplate
        # neighbors = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        # def h(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])

        # open_set = [(h(start,goal), start)]
        # came_from = {}
        # g_score = {start: 0}
        # closed = set()

        # while open_set:
        #     _, curr = heapq.heappop(open_set)
        #     if curr == goal:
        #         break
        #     if curr in closed:
        #         continue
        #     closed.add(curr)

        #     for dx,dy in neighbors:
        #         nb = (curr[0]+dx, curr[1]+dy)
        #         # bounds check
        #         if not (0 <= nb[0] < grid.shape[1] and 0 <= nb[1] < grid.shape[0]):
        #             continue
        #         c = cell_cost(*nb)
        #         if c is None:
        #             continue
        #         # diagonal penalty
        #         step = c * (1.414 if dx and dy else 1.0)
        #         tentative_g = g_score[curr] + step
        #         if tentative_g < g_score.get(nb, float('inf')):
        #             g_score[nb] = tentative_g
        #             f = tentative_g + h(nb, goal)
        #             heapq.heappush(open_set, (f, nb))
        #             came_from[nb] = curr

        # # 6) reconstruct & publish
        # if goal not in came_from:
        #     rospy.logerr("A* failed to find a path")
        #     return

        # path_idx = []
        # node = goal
        # while node != start:
        #     path_idx.append(node)
        #     node = came_from[node]
        # path_idx.append(start)
        # path_idx.reverse()

        # # convert to world + publish
        # self.publish_semantic_path(path_idx)
        # self.global_path = [ (p.pose.position.x, p.pose.position.y)
        #                     for p in self.last_published_path.poses ]
        # rospy.loginfo(f"Custom A* → {len(path_idx)} waypoints")

