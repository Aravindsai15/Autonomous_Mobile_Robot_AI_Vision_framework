#!/usr/bin/env python3


# #!/usr/bin/env python3
# import time
# import numpy as np
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import NavSatFix
# from nav_msgs.msg import OccupancyGrid, Path
# from geometry_msgs.msg import PoseStamped, Point, Quaternion, Pose, Twist
# from std_msgs.msg import Bool, Header
# from geodesy.utm import fromLatLong
# import heapq
# from collections import deque
# from tf_transformations import euler_from_quaternion
# # 
# # 

# class NavigationManager(Node):
#     def __init__(self):
#         super().__init__('navigation_manager')
        
#         # Parameters
#         self.declare_parameter('gps_timeout', 5.0)  # seconds
#         self.declare_parameter('local_costmap_topic', '/semantic_map')
#         self.declare_parameter('goal_tolerance', 0.5)  # meters
#         self.declare_parameter('assumed_speed', 0.5)  # m/s
#         self.declare_parameter('max_dead_reckoning_distance', 50.0)  # meters
        
#         # Subscribers
#         self.create_subscription(NavSatFix, '/gps/filtered', self.gps_callback, 10)
#         self.create_subscription(OccupancyGrid, 
#                                self.get_parameter('local_costmap_topic').value,
#                                self.semantic_callback, 10)
#         self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
#         self.create_subscription(PoseStamped, '/localization/filtered_odom', 
#                                self.odom_callback, 10)

#         # Publishers
#         self.path_pub = self.create_publisher(Path, '/global_path', 10)
#         self.mode_pub = self.create_publisher(Bool, '/nav_mode', 10)
#         self.cmd_vel_pub = self.create_publisher(Twist, '/nav_vel', 10)
        
#         # State
#         self.gps_active = True
#         self.last_gps_time = None
#         self.current_gps = None
#         self.semantic_costmap = None
#         self.target_pose = None
#         self.current_pose = None
#         self.global_path = []
#         self.current_waypoint_idx = 0
#         self.distance_covered = 0.0
#         self.last_position = None
#         self.start_time = None
        
#         # Timer for control loop
#         self.create_timer(0.1, self.navigation_loop)
#         self.create_timer(1.0, self.update_navigation_mode)

#     def gps_callback(self, msg):
#         self.last_gps_time = self.get_clock().now()
#         self.current_gps = msg

#     def odom_callback(self, msg):
#         self.current_pose = msg
#         if self.last_position is not None:
#             # Update distance covered based on position change
#             dx = msg.pose.position.x - self.last_position.x
#             dy = msg.pose.position.y - self.last_position.y
#             self.distance_covered += np.sqrt(dx**2 + dy**2)
#         self.last_position = msg.pose.position

#     def semantic_callback(self, msg):
#         self.semantic_costmap = msg

#     def goal_callback(self, msg):
#         self.target_pose = msg
#         if self.current_gps:
#             self.start_time = self.get_clock().now()
#             self.distance_covered = 0.0
#             self.plan_path()


#     def world_to_grid(self, world_pos, map_info):
#         """Convert world coordinates to grid indices"""
#         return (
#             int((world_pos.x - map_info.origin.position.x) / map_info.resolution),
#             int((world_pos.y - map_info.origin.position.y) / map_info.resolution)
#         )


#     def update_navigation_mode(self):
#         gps_timeout = self.get_parameter('gps_timeout').value
#         gps_valid = bool(self.last_gps_time and 
#                         (self.get_clock().now() - self.last_gps_time).nanoseconds < gps_timeout * 1e9)
        
#         if self.gps_active and not gps_valid:
#             self.gps_active = False
#             self.get_logger().warn("Switching to dead reckoning mode")
            
#         mode_msg = Bool()
#         mode_msg.data = self.gps_active
#         self.mode_pub.publish(mode_msg)

#     def plan_path(self):
#         if not self.target_pose or not self.semantic_costmap:
#             return
            
#         # Convert target to grid coordinates
#         goal_grid = self.world_to_grid(self.target_pose.pose.position, 
#                                      self.semantic_costmap.info)
        
#         # Get start position
#         if self.gps_active and self.current_gps:
#             start_pose = self.convert_gps_to_map(self.current_gps)
#         elif self.current_pose:
#             start_pose = self.current_pose
#         else:
#             return
            
#         start_grid = self.world_to_grid(start_pose.pose.position,
#                                       self.semantic_costmap.info)
        
#         # Generate path
#         path = self.a_star_planner(self.semantic_costmap, start_grid, goal_grid)
#         if path:
#             self.global_path = path
#             self.current_waypoint_idx = 0
#             self.publish_path(path, self.semantic_costmap.info)

#     def navigation_loop(self):
#         if not self.global_path or not self.current_pose:
#             return
            
#         # Check if reached goal
#         current_pos = self.current_pose.pose.position
#         goal_pos = self.target_pose.pose.position
#         distance_to_goal = np.sqrt((current_pos.x - goal_pos.x)**2 + 
#                                  (current_pos.y - goal_pos.y)**2)
        
#         if distance_to_goal < self.get_parameter('goal_tolerance').value:
#             self.stop_robot()
#             return
            
#         # Check dead reckoning limits
#         if not self.gps_active and (self.distance_covered >= 
#                                    self.get_parameter('max_dead_reckoning_distance').value):
#             self.stop_robot()
#             self.get_logger().error("Max dead reckoning distance reached")
#             return
        
#             # Stuck recovery
#         if self.check_stuck():
#             self.recover_from_stuck()
#             self.plan_path()  # Replan after recovery
#             return
        
#         # Dynamic replanning every 2 seconds
#         if self.time_since_last_plan > 2.0:
#             self.plan_path()
#             self.time_since_last_plan = 0.0
        

#         # Follow the path
#         self.follow_path()

#     def follow_path(self):
#         if not self.global_path or self.current_waypoint_idx >= len(self.global_path):
#             self.stop_robot()
#             return

#         current_pos = self.current_pose.pose.position
#         waypoint = self.global_path[self.current_waypoint_idx]
        
#         # Convert waypoint to world coordinates
#         map_info = self.semantic_costmap.info
#         wp_world = Point()
#         wp_world.x = waypoint[0] * map_info.resolution + map_info.origin.position.x
#         wp_world.y = waypoint[1] * map_info.resolution + map_info.origin.position.y
        
#         # Check if reached current waypoint
#         distance_to_wp = np.sqrt((current_pos.x - wp_world.x)**2 + 
#                             (current_pos.y - wp_world.y)**2)
        
#         if distance_to_wp < 0.3:  # Waypoint reached
#             self.current_waypoint_idx += 1
#             return
            
#         # Adjust speed based on terrain
#         waypoint_cost = self.semantic_costmap.data[
#             waypoint[1] * map_info.width + waypoint[0]]
#         max_speed = 0.5 if waypoint_cost == 0 else 0.2  # Slower on sidewalk
        
#         # Simple proportional control
#         linear_speed = min(max_speed, distance_to_wp * 0.5)
#         angle_to_wp = np.arctan2(wp_world.y - current_pos.y, 
#                             wp_world.x - current_pos.x)
        
#         # Get current yaw from quaternion
#         _, _, yaw = self.quaternion_to_euler(self.current_pose.pose.orientation)
#         angle_error = self.normalize_angle(angle_to_wp - yaw)
        
#         cmd_vel = Twist()
#         cmd_vel.linear.x = linear_speed
#         cmd_vel.angular.z = angle_error * 0.5
        
#         self.cmd_vel_pub.publish(cmd_vel)

#     def stop_robot(self):
#         cmd_vel = Twist()
#         self.cmd_vel_pub.publish(cmd_vel)

#     def a_star_planner(self, costmap, start, goal):
#         grid = np.array(costmap.data).reshape(
#             costmap.info.height,
#             costmap.info.width)
        
#         # Priority queue: (f_cost, g_cost, current, path)
#         open_set = [(0 + self.heuristic(start, goal), 0, start, [start])]
#         closed_set = set()
        
#         while open_set:
#             _, g_cost, current, path = heapq.heappop(open_set)
            
#             if current == goal:
#                 return path
                
#             if current in closed_set:
#                 continue
#             closed_set.add(current)
            
#             for neighbor in self.get_neighbors(current, grid.shape):
#                 # Hard obstacle (100) - completely blocked
#                 if grid[neighbor[1]][neighbor[0]] > 50:
#                     continue
                    
#                 # Cost penalty: prefer road (0) over sidewalk (50)
#                 terrain_cost = 0 if grid[neighbor[1]][neighbor[0]] == 0 else 10
                
#                 # Directional bias at junctions
#                 bias = 0
#                 if self.near_junction(current, grid):
#                     goal_dx = goal[0] - current[0]
#                     if goal_dx < 0:  # Goal is left
#                         bias = -10 if neighbor[0] < current[0] else 10
#                     else:  # Goal is right
#                         bias = 10 if neighbor[0] < current[0] else -10
                
#                 new_g = g_cost + 1 + terrain_cost + bias
#                 new_f = new_g + self.heuristic(neighbor, goal)
#                 heapq.heappush(open_set, (new_f, new_g, neighbor, path + [neighbor]))
        
#         self.get_logger().warn("A* failed to find path!")
#         return None

#     def near_junction(self, cell, grid):
#         """Improved junction detection (4-connected)"""
#         road_neighbors = 0
#         for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
#             nx, ny = cell[0] + dx, cell[1] + dy
#             if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]:
#                 if grid[ny][nx] <= 50:  # Road or sidewalk
#                     road_neighbors += 1
#         return road_neighbors >= 3

#     def check_stuck(self):
#         """Check if robot is stuck in sidewalk"""
#         if not self.current_pose:
#             return False
        
#         current_grid = self.world_to_grid(
#             self.current_pose.pose.position,
#             self.semantic_costmap.info
#         )
#         idx = current_grid[1] * self.semantic_costmap.info.width + current_grid[0]
#         return self.semantic_costmap.data[idx] == 50  # Stuck in sidewalk

#     def recover_from_stuck(self):
#         """Execute recovery maneuver"""
#         self.get_logger().warn("Recovering from stuck position!")
#         cmd_vel = Twist()
#         cmd_vel.linear.x = -0.2  # Reverse
#         cmd_vel.angular.z = 0.5  # Turn
#         for _ in range(10):  # Try for 1 second (10 * 0.1s)
#             self.cmd_vel_pub.publish(cmd_vel)
#             time.sleep(0.1)
#         self.stop_robot()

#     def convert_gps_to_map(self, gps_msg):
#         try:
#             utm_point = fromLatLong(gps_msg.latitude, gps_msg.longitude)
#             pose = PoseStamped()
#             pose.header = Header(frame_id='map', stamp=self.get_clock().now().to_msg())
#             pose.pose.position = Point(x=utm_point.easting, y=utm_point.northing, z=0.0)
#             pose.pose.orientation = Quaternion(w=1.0)
#             return pose
#         except Exception as e:
#             self.get_logger().error(f"GPS conversion failed: {str(e)}")
#             return None



#     def heuristic(self, a, b):
#         return abs(a[0] - b[0]) + abs(a[1] - b[1])

#     def get_neighbors(self, cell, grid_shape):
#         neighbors = []
#         for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
#             nx, ny = cell[0] + dx, cell[1] + dy
#             if 0 <= nx < grid_shape[1] and 0 <= ny < grid_shape[0]:
#                 neighbors.append((nx, ny))
#         return neighbors

#     def quaternion_to_euler(self, q):
#         # Convert quaternion to Euler angles (roll, pitch, yaw)
#         x = q.x
#         y = q.y
#         z = q.z
#         w = q.w
        
#         t0 = +2.0 * (w * x + y * z)
#         t1 = +1.0 - 2.0 * (x * x + y * y)
#         roll = np.arctan2(t0, t1)
        
#         t2 = +2.0 * (w * y - z * x)
#         t2 = +1.0 if t2 > +1.0 else t2
#         t2 = -1.0 if t2 < -1.0 else t2
#         pitch = np.arcsin(t2)
        
#         t3 = +2.0 * (w * z + x * y)
#         t4 = +1.0 - 2.0 * (y * y + z * z)
#         yaw = np.arctan2(t3, t4)
        
#         return roll, pitch, yaw

#     def normalize_angle(self, angle):
#         # Normalize angle to [-pi, pi]
#         while angle > np.pi:
#             angle -= 2 * np.pi
#         while angle < -np.pi:
#             angle += 2 * np.pi
#         return angle

#     def publish_path(self, grid_path, map_info):
#         path_msg = Path()
#         path_msg.header = Header(frame_id='map', stamp=self.get_clock().now().to_msg())
        
#         for cell in grid_path:
#             pose = PoseStamped()
#             pose.header = path_msg.header
#             pose.pose.position.x = cell[0] * map_info.resolution + map_info.origin.position.x
#             pose.pose.position.y = cell[1] * map_info.resolution + map_info.origin.position.y
#             pose.pose.orientation.w = 1.0
#             path_msg.poses.append(pose)
            
#         self.path_pub.publish(path_msg)

# def main(args=None):
#     rclpy.init(args=args)
#     node = NavigationManager()
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
import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Pose, Twist
from std_msgs.msg import Bool, Header
from geodesy.utm import fromLatLong
import heapq
from collections import deque
from tf_transformations import euler_from_quaternion


class NavigationManager(Node):
    def __init__(self):
        super().__init__('navigation_manager')
        
        # Parameters
        self.declare_parameter('gps_timeout', 5.0)  # seconds
        self.declare_parameter('local_costmap_topic', '/semantic_map')
        self.declare_parameter('goal_tolerance', 0.5)  # meters
        self.declare_parameter('assumed_speed', 0.5)  # m/s
        self.declare_parameter('max_dead_reckoning_distance', 50.0)  # meters
        
        # Subscribers
        self.create_subscription(NavSatFix, '/gps/filtered', self.gps_callback, 10)
        self.create_subscription(OccupancyGrid, 
                               self.get_parameter('local_costmap_topic').value,
                               self.semantic_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.create_subscription(PoseStamped, '/localization/filtered_odom', 
                               self.odom_callback, 10)

        # Publishers
        self.path_pub = self.create_publisher(Path, '/global_path', 10)
        self.mode_pub = self.create_publisher(Bool, '/nav_mode', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/nav_vel', 10)
        
        # State
        self.gps_active = True
        self.last_gps_time = None
        self.current_gps = None
        self.semantic_costmap = None
        self.target_pose = None
        self.current_pose = None
        self.global_path = []
        self.current_waypoint_idx = 0
        self.distance_covered = 0.0
        self.last_position = None
        self.start_time = self.get_clock().now()
        self.time_since_last_plan = 0.0  # Added missing variable
        
        # Timer for control loop
        self.create_timer(0.1, self.navigation_loop)
        self.create_timer(1.0, self.update_navigation_mode)

    def gps_callback(self, msg):
        self.last_gps_time = self.get_clock().now()
        self.current_gps = msg

    def odom_callback(self, msg):
        self.current_pose = msg
        if self.last_position is not None:
            dx = msg.pose.position.x - self.last_position.x
            dy = msg.pose.position.y - self.last_position.y
            self.distance_covered += np.hypot(dx, dy)
        self.last_position = msg.pose.position

    def semantic_callback(self, msg):
        self.semantic_costmap = msg


    def goal_callback(self, msg):
        self.target_pose = msg
        if self.current_gps or self.current_pose:
            self.start_time = self.get_clock().now()
            self.distance_covered = 0.0
            self.plan_path()


    def world_to_grid(self, world_pos, map_info):
        """Safe grid conversion with bounds checking"""
        x = int((world_pos.x - map_info.origin.position.x) / map_info.resolution)
        y = int((world_pos.y - map_info.origin.position.y) / map_info.resolution)
        return (
            np.clip(int(x), 0, map_info.width-1),
            np.clip(int(y), 0, map_info.height-1)
            )


    def update_navigation_mode(self):
        gps_timeout = self.get_parameter('gps_timeout').value
        if self.last_gps_time:
            elapsed = (self.get_clock().now() - self.last_gps_time).nanoseconds * 1e-9
            self.gps_active = elapsed < gps_timeout
        else:
            self.gps_active = False
            
        mode_msg = Bool()
        mode_msg.data = self.gps_active
        self.mode_pub.publish(mode_msg)
        

    def plan_path(self):
        if not self.target_pose or not self.semantic_costmap:
            return
            
        # Get start position
        start_pose = None
        if self.gps_active and self.current_gps:
            start_pose = self.convert_gps_to_map(self.current_gps)
        elif self.current_pose:
            start_pose = self.current_pose
            
        if not start_pose:
            self.get_logger().warn("No valid start position for planning!")
            return
            
        # Convert to grid coordinates
        map_info = self.semantic_costmap.info
        start_grid = self.world_to_grid(start_pose.pose.position, map_info)
        goal_grid = self.world_to_grid(self.target_pose.pose.position, map_info)
        
        # Generate path
        path = self.a_star_planner(self.semantic_costmap, start_grid, goal_grid)
        if path:
            self.global_path = path
            self.current_waypoint_idx = 0
            self.publish_path(path, map_info)
            self.time_since_last_plan = 0.0

    def navigation_loop(self):
        if not self.global_path or not self.current_pose:
            return
            
        # Update timing
        now = self.get_clock().now()
        self.time_since_last_plan += (now - self.start_time).nanoseconds * 1e-9
        self.start_time = now
        
        # Goal check
        current_pos = self.current_pose.pose.position
        goal_pos = self.target_pose.pose.position
        distance_to_goal = np.hypot(current_pos.x - goal_pos.x, current_pos.y - goal_pos.y)
        
        if distance_to_goal < self.get_parameter('goal_tolerance').value:
            self.stop_robot()
            return
            
        # Dead reckoning check
        if not self.gps_active and (self.distance_covered >= 
                                   self.get_parameter('max_dead_reckoning_distance').value):
            self.stop_robot()
            self.get_logger().error("Max dead reckoning distance reached")
            return
            
        # Dynamic replanning
        if self.time_since_last_plan > 2.0:
            self.plan_path()
            
        # Follow path
        self.follow_path()

    def follow_path(self):
        if not self.global_path or self.current_waypoint_idx >= len(self.global_path):
            self.stop_robot()
            return

        current_pos = self.current_pose.pose.position
        waypoint = self.global_path[self.current_waypoint_idx]
        
        # Convert waypoint to world coordinates
        map_info = self.semantic_costmap.info
        wp_world = Point()
        wp_world.x = waypoint[0] * map_info.resolution + map_info.origin.position.x
        wp_world.y = waypoint[1] * map_info.resolution + map_info.origin.position.y
        
        # Calculate distance to waypoint
        distance_to_wp = np.hypot(current_pos.x - wp_world.x, current_pos.y - wp_world.y)
        
        if distance_to_wp < 0.3:  # Waypoint reached
            self.current_waypoint_idx += 1
            return
            
        # Calculate orientation error
        angle_to_wp = np.arctan2(wp_world.y - current_pos.y, wp_world.x - current_pos.x)
        current_yaw = self.quaternion_to_euler(self.current_pose.pose.orientation)
        angle_error = self.normalize_angle(angle_to_wp - current_yaw)
        
        # Generate velocity command
        cmd_vel = Twist()
        cmd_vel.linear.x = min(0.5, distance_to_wp * 0.5)  # Max speed 0.5 m/s
        cmd_vel.angular.z = angle_error * 0.5  # Proportional control
        
        self.cmd_vel_pub.publish(cmd_vel)

    def quaternion_to_euler(self, q):
        """Convert quaternion to yaw using TF"""
        return euler_from_quaternion([q.x, q.y, q.z, q.w])[2]




#-------------------------------------------------------------------------




    def stop_robot(self):
        cmd_vel = Twist()
        self.cmd_vel_pub.publish(cmd_vel)

    def a_star_planner(self, costmap, start, goal):
        grid = np.array(costmap.data).reshape(costmap.info.height, costmap.info.width)
        
        open_set = [(0 + self.heuristic(start, goal), 0, start, [start])]
        closed_set = set()
        
        while open_set:
            _, g_cost, current, path = heapq.heappop(open_set)
            
            if current == goal:
                return self.smooth_path(grid, path)  # Apply path smoothing
            
            if current in closed_set:
                continue
            closed_set.add(current)
            
            for neighbor in self.get_neighbors(current, grid.shape):
                if grid[neighbor[1]][neighbor[0]] >= self.OBSTACLE_COST:
                    continue
                    
                # Cost calculation
                move_cost = 1.4 if abs(neighbor[0]-current[0]) + abs(neighbor[1]-current[1]) > 1 else 1.0
                terrain_cost = 0 if grid[neighbor[1]][neighbor[0]] == self.ROAD_COST else 10
                
                # Directional bias
                bias = 0
                if self.near_junction(current, grid):
                    goal_vec = (goal[0]-current[0], goal[1]-current[1])
                    neighbor_vec = (neighbor[0]-current[0], neighbor[1]-current[1])
                    bias = -10 if (goal_vec[0]*neighbor_vec[1] - goal_vec[1]*neighbor_vec[0]) > 0 else 10
                
                new_g = g_cost + move_cost + terrain_cost + bias
                new_f = new_g + self.heuristic(neighbor, goal)
                heapq.heappush(open_set, (new_f, new_g, neighbor, path + [neighbor]))
        
        self.get_logger().warn("A* failed to find path!")
        return None

    def near_junction(self, cell, grid):
        """Improved junction detection (4-connected)"""
        road_neighbors = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cell[0] + dx, cell[1] + dy
            if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]:
                if grid[ny][nx] <= 50:  # Road or sidewalk
                    road_neighbors += 1
        return road_neighbors >= 3

    def check_stuck(self):
        """Check if robot is stuck in sidewalk"""
        if not self.current_pose:
            return False
        
        current_grid = self.world_to_grid(
            self.current_pose.pose.position,
            self.semantic_costmap.info
        )
        idx = current_grid[1] * self.semantic_costmap.info.width + current_grid[0]
        return self.semantic_costmap.data[idx] == 50  # Stuck in sidewalk

    def recover_from_stuck(self):
        """Execute recovery maneuver"""
        self.get_logger().warn("Recovering from stuck position!")
        cmd_vel = Twist()
        cmd_vel.linear.x = -0.2  # Reverse
        cmd_vel.angular.z = 0.5  # Turn
        for _ in range(10):  # Try for 1 second (10 * 0.1s)
            self.cmd_vel_pub.publish(cmd_vel)
            time.sleep(0.1)
        self.stop_robot()

    def convert_gps_to_map(self, gps_msg):
        try:
            utm_point = fromLatLong(gps_msg.latitude, gps_msg.longitude)
            pose = PoseStamped()
            pose.header = Header(frame_id='map', stamp=self.get_clock().now().to_msg())
            pose.pose.position = Point(x=utm_point.easting, y=utm_point.northing, z=0.0)
            pose.pose.orientation = Quaternion(w=1.0)
            return pose
        except Exception as e:
            self.get_logger().error(f"GPS conversion failed: {str(e)}")
            return None



    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, cell, grid_shape):
        neighbors = []
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
            nx, ny = cell[0] + dx, cell[1] + dy
            if 0 <= nx < grid_shape[1] and 0 <= ny < grid_shape[0]:
                neighbors.append((nx, ny))
        return neighbors

    def quaternion_to_euler(self, q):
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        x = q.x
        y = q.y
        z = q.z
        w = q.w
        
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)
        
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = np.arcsin(t2)
        
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)
        
        return roll, pitch, yaw

    def normalize_angle(self, angle):
        # Normalize angle to [-pi, pi]
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def publish_path(self, grid_path, map_info):
        path_msg = Path()
        path_msg.header = Header(frame_id='map', stamp=self.get_clock().now().to_msg())
        
        for cell in grid_path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = cell[0] * map_info.resolution + map_info.origin.position.x
            pose.pose.position.y = cell[1] * map_info.resolution + map_info.origin.position.y
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
            
        self.path_pub.publish(path_msg)



def main(args=None):
    rclpy.init(args=args)
    node = NavigationManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()