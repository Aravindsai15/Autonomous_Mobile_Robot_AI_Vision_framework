
# Artificial Intelligence-Driven Vision Framework for Real-Time Navigation of Autonomous Mobile Robots

## Introduction
This thesis presents an advanced AI-driven vision framework designed for real-time navigation of autonomous mobile robots. The framework integrates state-of-the-art computer vision algorithms with deep learning models to enable robust perception, obstacle avoidance, path planning, and environmental understanding in dynamic environments. Built on ROS 1 (Melodic Morenia) and leveraging modern AI techniques, this system provides a comprehensive solution for autonomous navigation across various robotic platforms, facilitates autonomous navigation without prebuilt maps, and incorporates redundant sensor / functionality to tackle robot movement during GPS loss.


## System Architecture Overview

The framework employs a modular architecture with the following core components:

- **Perception Module**: RealSense ROS integration for multi-sensor data acquisition and processing
- **AI Processing Core**: Deep learning models for object detection and scene understanding
- **Localization Core**: Multi-sensor odometry for robust positioning and redundant sensor integration in GPS-denied environments.
- **Navigation Stack**: Real-time path planning using A* and DWA (Dynamic Window Approach Planner)
- **Control Interface**: Unified API for robot motion control


### Sensors
- **RGB-D + Stereo Camera** (Intel RealSense D435i or similar)
- **IMU**: 9-axis Abs orientation sensor (BNO055)
   -    3-axis Accelerometer
   -    3-axis Gyroscope
   -    3-axis Magnetometer
   -    On-chip sensor fusion with Compass (North direction aligned with X axis)
- **Beitian BN – 880 GNSS Module**
- **Computing Unit**: NVIDIA Jetson Nano A57
- **Arduino Mega 2560**

##  System Specifications

### Hardware & OS Environment
- Operating System: Ubuntu 18.04.6 LTS (Bionic Beaver)
- Platform: NVIDIA Jetson (JetPack compatible)
- CUDA Version: 10.2.300
- GPU Acceleration: NVIDIA CUDA-enabled GPU

### Software Framework
- ROS Distribution: ROS 1 Melodic Morenia
- Python Versions:
    - Python 2.7.17 (ROS Melodic compatibility)
    - Python 3.6.9 (to retrieve hardware integrated CUDA for AI/ML processing)
- AI Framework: PyTorch 1.8.0
- Numerical Computing: NumPy 1.19.4
- Computer Vision: OpenCV (ROS Melodic bundled)
- Visualization: RViz
- Point Cloud Library: PCL 1.8 (ROS Melodic)

<img width="1091" height="522" alt="AMR specs" src="https://github.com/user-attachments/assets/a5845922-a42d-41d8-a16b-f169ec12a6b4" />

## 1. Clone the Repository
### Option 1: Clone via GitHub CLI
`gh repo clone Aravindsai15/Autonomous_Mobile_Robot_AI_Vision_framework  `

### Option 2: Clone via Git
`git clone https://github.com/Aravindsai15/Autonomous_Mobile_Robot_AI_Vision_framework.git`


or download manually and place it inside your ROS workspace:

~/autobot_ai_ws/src/

## 2. Build the Package
```bash
cd ~/autobot_ai_ws
catkin_make
source devel/setup.bash
```

## Visual Reference of repository structure and project subfolders
<img width="1600" height="1001" alt="tree_1" src="https://github.com/user-attachments/assets/9ee4c386-5a89-4bc5-ae73-715d74d7cb23" />
<img width="1579" height="956" alt="tree_2" src="https://github.com/user-attachments/assets/1c269ee7-8ca3-4357-ad6d-8942d5636049" />

## Concised view of repository structure
<img width="1604" height="973" alt="tree_4" src="https://github.com/user-attachments/assets/892fb275-a44e-4fa0-b2e0-b7178916ee45" />


## 3. Source the Workspace

After building the workspace, source the environment properly to overlay your custom ROS package on top of your system installation:

### Set up ROS Melodic environment
`source /opt/ros/melodic/setup.bash`

### Go to your workspace
`cd ~/autobot_ai_ws/`

### Source your workspace setup
`source devel/setup.bash`


(Tip: You can add these lines to your ~/.bashrc file to make sourcing automatic.)

## 4. Run the Command to Initiate and Operate the AMR

### Launch the Autonomous Mobile Robot (AMR) framework using the ROS launch command:

```cd ~/autobot_ai_ws/src/
roslaunch autobot_ai autobot_ai.launch.xml
```

<img width="1028" height="673" alt="initiate_AMR" src="https://github.com/user-attachments/assets/f5de88ee-de23-4eb5-bf85-93c7f3211878" />

This command initializes all core nodes, including:
AI vision and perception modules
Navigation and planning stack
Sensor fusion and localization
Motor driver control
Make sure your workspace has been built and sourced before running the launch command (see Step 3).

## 5. Open RViz and Set the Reference Frame
Once RViz launches automatically (or if you open it manually with rosrun rviz rviz), set the Fixed Frame under the “Global Options” section to:
- odom → for odometry-based visualization (world-relative)
- base_link → for robot-centric visualization
This ensures all topics (e.g., /map, /odom, /tf) align correctly in the visualization.

## 6. Visualize Data in RViz
To visualize the robot’s perception, planning, and navigation data in RViz, import and enable the following topics:
| **Topic Name**                  | **Description / Notes**                                             |
| ------------------------------- | ------------------------------------------------------------------- |
| `/camera/color/image_raw`       | Raw RGB camera feed from onboard vision sensor                      |
| `/segmentation_output/image`    | AI-based semantic segmentation output                               |
| `/colored_segmentation/output`  | Colorized segmentation overlay for better visualization             |
| `/semantic_costmap`             | Costmap showing semantic layers for obstacle avoidance              |
| `/odometry/filtered`            | Fused odometry (disable *Covariance* display by unchecking its box) |
| `/move_base/GlobalPlanner/plan` | Global navigation path generated by Move Base                       |
| `/move_base_simple/goal`        | Goal pose issued manually from RViz (*2D Nav Goal*)                 |


<img width="1566" height="975" alt="rviz topic" src="https://github.com/user-attachments/assets/666b7edb-4c15-40be-8b55-86004d3dfd7d" />


## 7. Issuing a Goal Pose and Checking Wheel Motion
To command the robot to navigate to a desired location:
 In RViz, locate the 2D Nav Goal tool on the top toolbar (usually represented by a red arrow icon).
 Click on the map area to define the goal position, then drag slightly to set the desired orientation.
 Once the goal is issued, the robot will begin path planning and motion execution.
 Observe the wheel rotation 


**Note for Supervisors, readers and developers**: This framework represents a complete pipeline from perception to control, with particular emphasis on real-time performance and adaptability to different robotic platforms. The modular architecture allows for easy extension and modification of individual components while maintaining system integrity.

For any questions, suggestions, or further discussions, please feel free to contact me:
Email: [aravindsai02@gmail.com]

I am happy to provide guidance, consult on integration, or discuss potential improvements and extensions.
