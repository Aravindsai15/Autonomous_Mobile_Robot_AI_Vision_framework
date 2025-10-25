#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced NavigationManager with state management
- Added active goal tracking
- Improved condition checks for replanning
- Fixed goal clearing after completion
"""
import rospy
import math
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Twist, Pose
from std_msgs.msg import Header
from sensor_msgs.msg import NavSatFix
from navfn.srv import MakeNavPlan, MakeNavPlanRequest
from pyproj import Proj

class NavigationManager:
    def __init__(self):
        rospy.init_node('navigation_manager')
            # Add explicit frame_id parameters
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')
        self.base_frame = rospy.get_param('~base_frame', 'base_link')
        self.global_frame = rospy.get_param('~global_frame', 'map')
        self.current_pose_header = Header(frame_id=self.odom_frame)

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
        self.has_active_goal = False  # Track if we have an active goal
        self.initial_pose_received = False  # Track if we have initial pose

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

        # Control loop
        rospy.Timer(rospy.Duration(0.1), self.control_loop)
        rospy.loginfo("NavigationManager ready (using semantic-aware A*)")

    def rviz_goal_cb(self, msg):
        if self.use_gps_goals:
            rospy.logwarn("Ignoring RViz goal in GPS mode")
            return
        self.goal_utm = (msg.pose.position.x, msg.pose.position.y)
        self.has_active_goal = True
        rospy.loginfo("RViz goal set: %.2f, %.2f", *self.goal_utm)
        self.plan_path()

    def gps_goal_cb(self, msg):
        lat, lon = msg.pose.position.x, msg.pose.position.y
        try:
            e, n = self.utm_proj(lon, lat)
            self.goal_utm = (e, n)
            self.has_active_goal = True
            rospy.loginfo("GPS goal set: UTM(%.2f, %.2f)", e, n)
            self.plan_path()
        except Exception as e:
            rospy.logerr("GPS->UTM conversion failed: %s", e)

    def gps_status_cb(self, msg):
        # Optional: monitor GPS health
        pass

    def odom_cb(self, msg):
        """More robust odometry handling"""
        if msg.header.frame_id != self.odom_frame:
            rospy.logwarn_once(f"Odometry frame mismatch: expected {self.odom_frame}, got {msg.header.frame_id}")
        
        self.current_pose = msg.pose.pose
        self.current_pose_header = msg.header  # Store full header
        
        if not self.initial_pose_received:
            rospy.loginfo("Initial pose received in frame: %s", msg.header.frame_id)
            self.initial_pose_received = True

    def semantic_callback(self, msg):
        """Process semantic costmap with proper value scaling"""
        self.costmap = msg
        
        width = msg.info.width
        height = msg.info.height
        rospy.loginfo("Semantic costmap size (cells): %dx%d, total: %d", width, height, width*height)
        

    def validate_path(self, path):
        if not self.costmap:
            rospy.logwarn("No costmap for validation")
            return False
        origin = self.costmap.info.origin.position
        res = self.costmap.info.resolution
        w = self.costmap.info.width
        h = self.costmap.info.height
        for x, y in path:
            mx = int((x - origin.x) / res)
            my = int((y - origin.y) / res)
            if not (0 <= mx < w and 0 <= my < h):
                rospy.logwarn("Path point out of costmap bounds")
                return False
            idx = my * w + mx
            cost = self.costmap.data[idx]
            if cost == -1 or cost >= 50:
                rospy.logwarn("Path intersects non-drivable area at (%d, %d)", mx, my)
                return False
        return True

    def plan_path(self):
        if self.current_pose_header.frame_id != self.global_frame:
            rospy.logwarn(f"Cannot plan - pose in {self.current_pose_header.frame_id} but need {self.global_frame}")
            return False

        # Check if we have all required information
        if not self.has_active_goal:
            rospy.logwarn("Skipping planning: No active goal")
            return False
            
        if not self.wait_for_costmap():
            rospy.logerr("Costmap not available")
            return False
            
        if not self.initial_pose_received:
            rospy.logwarn("Skipping planning: Initial pose not received")
            return False

        # Prepare request
        req = MakeNavPlanRequest()
        req.start = PoseStamped(
            header=Header(stamp=rospy.Time.now(), frame_id="map"),
            pose=self.current_pose
        )
        goal_pose = Pose()
        goal_pose.position.x, goal_pose.position.y = self.goal_utm
        goal_pose.orientation = Quaternion(0, 0, 0, 1)
        req.goal = PoseStamped(
            header=Header(stamp=rospy.Time.now(), frame_id="map"),
            pose=goal_pose
        )
        try:
            resp = self.plan_srv(req)
            if not resp.plan.poses:
                rospy.logwarn("Empty plan from service")
                return False
            pts = [(p.pose.position.x, p.pose.position.y) for p in resp.plan.poses]
            if not self.validate_path(pts):
                rospy.logwarn("Validation failed, recovering")
                return self.recover_and_replan()
            self.original_path = pts
            self.global_path = list(pts)
            self.current_wp_idx = 0
            self.last_plan_time = rospy.Time.now()
            self.publish_path(self.global_path)
            rospy.loginfo("Planned path with %d points", len(self.global_path))
            return True
        except rospy.ServiceException as e:
            rospy.logerr("Plan service error: %s", e)
            return False

    def recover_and_replan(self):
        self.stop_robot()
        cmd = Twist(); cmd.linear.x = -0.1; self.cmd_vel_pub.publish(cmd); rospy.sleep(0.5)
        cmd = Twist(); cmd.angular.z = 0.3; self.cmd_vel_pub.publish(cmd); rospy.sleep(1.0)
        return self.plan_path()

    def publish_path(self, path):
        msg = Path(); msg.header = Header(stamp=rospy.Time.now(), frame_id="map")
        for x, y in path:
            ps = PoseStamped(); ps.header = msg.header; ps.pose.position = Point(x, y, 0); ps.pose.orientation = Quaternion(0, 0, 0, 1)
            msg.poses.append(ps)
        self.path_pub.publish(msg)

    def stop_robot(self):
        self.cmd_vel_pub.publish(Twist())

    def get_yaw(self, q):
        # Convert quaternion to yaw
        return math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

    def path_deviation(self):
        if not self.current_pose or not self.global_path:
            return 0
        cx, cy = self.current_pose.position.x, self.current_pose.position.y
        return min(math.hypot(mx-cx, my-cy) for mx, my in self.global_path)

    def wait_for_costmap(self, timeout=5.0):
        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time).to_sec() < timeout:
            if (self.costmap and self.costmap.info.resolution > 0 and
                len(self.costmap.data) > 0 and self.costmap.info.width > 0):
                return True
            rospy.sleep(0.1)
        return False


    def control_loop(self, event):
        # Skip if no active goal
        if not self.has_active_goal:
            return
            
        # Periodic replanning only if we have all required data
        if (rospy.Time.now() - self.last_plan_time).to_sec() > self.replan_interval:
            if self.initial_pose_received and self.costmap:
                rospy.loginfo("Periodic replanning triggered")
                self.plan_path()
            else:
                rospy.logwarn("Skipping replanning - missing pose or costmap")
                
        # Check if we have a valid path to follow
        if not self.global_path or self.current_wp_idx >= len(self.global_path):
            return
            
        # deviation check
        if self.path_deviation() > self.max_deviation:
            if self.initial_pose_received and self.costmap:
                rospy.logwarn("Deviation exceeded, replanning")
                self.plan_path()
            else:
                rospy.logwarn("Cannot replan - missing pose or costmap")
            return
            
        wx, wy = self.global_path[self.current_wp_idx]
        cx, cy = self.current_pose.position.x, self.current_pose.position.y
        yaw = self.get_yaw(self.current_pose.orientation)
        dist = math.hypot(wx-cx, wy-cy)
        angle = math.atan2(wy-cy, wx-cx)
        err = (angle - yaw + math.pi) % (2*math.pi) - math.pi
        
        # Move to next waypoint if close enough
        if dist < self.goal_tolerance:
            self.current_wp_idx += 1
            if self.current_wp_idx >= len(self.global_path):
                rospy.loginfo("Goal reached")
                self.stop_robot()
                hdr = Header(stamp=rospy.Time.now(), frame_id="GOAL_REACHED")
                self.status_pub.publish(hdr)
                # Clear active goal state
                self.has_active_goal = False
                self.goal_utm = None
                self.global_path = []
            return
            
        # Generate control command
        cmd = Twist()
        # Adaptive speed based on alignment
        if abs(err) > math.pi/4:  # >45° misalignment
            cmd.linear.x = 0.1
        else:
            cmd.linear.x = min(0.5, dist*0.5)
        # Angular control (proportional)
        cmd.angular.z = 1.0 * err
        self.cmd_vel_pub.publish(cmd)

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

