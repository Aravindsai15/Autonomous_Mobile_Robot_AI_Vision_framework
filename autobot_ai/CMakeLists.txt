cmake_minimum_required(VERSION 3.0.2)
project(autobot_ai)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  geometry_msgs
  nav_msgs
  cv_bridge
  image_transport
  move_base
  amcl
  map_server
  diagnostic_updater
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs geometry_msgs nav_msgs cv_bridge image_transport diagnostic_updater
)

catkin_install_python(
  PROGRAMS
    scripts/gpu_toggle_server.py
    scripts/amr_ai_vision_node.py
    scripts/navigation_manager.py
    scripts/goal_loc_publisher.py
    scripts/gps_simulated.py
    scripts/relay_camera_info_node.py
    scripts/imu_9axis_node.py
    scripts/odom_to_pose.py
    scripts/auto_navigation.py
    scripts/ekf_toggle.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY src/autobot_ai/
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)
