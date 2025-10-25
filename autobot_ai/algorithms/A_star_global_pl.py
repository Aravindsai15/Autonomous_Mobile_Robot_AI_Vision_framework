import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped

class AStarPlanner:
    def __init__(self):
        rospy.init_node("astar_planner")

        self.costmap = None
        self.costmap_sub = rospy.Subscriber("/semantic_costmap", OccupancyGrid, self.costmap_callback)
        self.path_pub = rospy.Publisher("/planned_path", Path, queue_size=1)

        # Simulated goal input
        self.goal = (90, 90)  # Example goal cell index

    def costmap_callback(self, msg):
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin = msg.info.origin

        data = np.array(msg.data, dtype=np.int8).reshape((self.height, self.width))
        self.costmap = data
        rospy.loginfo("Costmap received.")

        start = (self.width // 2, self.height // 2)  # center of map
        path = self.a_star(start, self.goal)
        self.publish_path(path)

    def a_star(self, start, goal):
        # Basic A* implementation here
        pass

    def publish_path(self, cell_path):
        ros_path = Path()
        ros_path.header.frame_id = "odom"
        ros_path.header.stamp = rospy.Time.now()

        for cell in cell_path:
            pose = PoseStamped()
            pose.header = ros_path.header
            pose.pose.position.x = self.origin.position.x + cell[0] * self.resolution
            pose.pose.position.y = self.origin.position.y + cell[1] * self.resolution
            pose.pose.orientation.w = 1.0
            ros_path.poses.append(pose)

        self.path_pub.publish(ros_path)

if __name__ == '__main__':
    AStarPlanner()
    rospy.spin()
