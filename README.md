# DisFlow: Scene Flow from Distance Field for Object Pose, Velocity Tracking, and Dynamic Object Reconstruction

DisFlow is a real-time framework for dynamic object perception built on top of the IDMP (Interactive Distance Field Mapping and Planning) framework.

Unlike conventional object tracking systems that only estimate object poses, DisFlow jointly performs:

- 6DoF object pose tracking
- Linear and angular velocity estimation
- Dynamic object reconstruction
- Surface normal estimation
- Distance and gradient field querying
- Uncertainty-aware modelling

The core idea is to represent an object using Gaussian Process Distance Field (GPDF) and estimate object motion through a novel **Distance Flow (DisFlow)** formulation derived from the temporal evolution of the distance field.

---

## Overview

DisFlow extends the IDMP framework from static scene mapping to dynamic object tracking and reconstruction.

The framework operates in an **object-centric reference frame**, where:

- the object geometry remains fixed,
- new observations are continuously fused,
- pose and velocity are estimated through distance-field registration.

---

## Framework Pipeline

```text
RGB-D Camera
      │
      ▼
Point Cloud
      │
      ▼
Distance Flow Registration
      │
      ├──► Pose Tracking
      │
      ├──► Velocity Tracking
      │
      ▼
Object Frame Transformation
      │
      ▼
GPDF Fusion
      │
      ├──► Distance Queries
      ├──► Gradient Queries
      ├──► Surface Normals
      ├──► Reconstruction
      └──► Uncertainty Estimation
```

---

## Relation to IDMP

DisFlow is built directly on top of the IDMP framework.

IDMP focuses on:

- Continuous distance field mapping
- Distance and gradient queries
- Human-robot collaboration
- Motion planning

DisFlow extends these capabilities to dynamic object perception by introducing:

| IDMP | DisFlow |
|--------|--------|
| Static scene mapping | Dynamic object tracking |
| Distance field fusion | Object-centric fusion |
| Distance queries | Distance flow estimation |
| Planning | Pose and velocity tracking |
| Human-robot interaction | Dynamic reconstruction |

---

## Dependencies

### Ubuntu

- Ubuntu 22.04
- ROS2 Humble
- Eigen3
- PCL
- OpenCV
- OpenMP (optional but recommended)

### RGB-D Sensors

Supported sensors include:

- Intel RealSense
- Azure Kinect
- ZED 2i

---

## Installation

Create a ROS2 workspace:

```bash
mkdir -p ~/disflow_ws/src
cd ~/disflow_ws/src
```

Clone the repository:

```bash
git clone https://github.com/LanWu076/disflow.git
```

Build:

```bash
cd ~/disflow_ws
colcon build --symlink-install
```

Source the workspace:

```bash
source /opt/ros/humble/setup.bash
source ~/disflow_ws/install/setup.bash
```

---

## Running DisFlow

### Terminal 1: Start RGB-D Camera

Example for RealSense:

```bash
ros2 launch realsense2_camera rs_pointcloud_launch.py
```

### Terminal 2: Run DisFlow

```bash
ros2 launch idmp_ros realsense.py
```

### Terminal 3: Visualisation

```bash
rviz2
```

---

## Published Topics

| Topic | Type | Description |
|---------|---------|---------|
| `/gp_pcl` | sensor_msgs/PointCloud2 | Accumulated object reconstruction point cloud |
| `/idmp/distance_flow` | sensor_msgs/PointCloud2 | Distance flow estimated from the GP distance field |
| `/idmp/gradient_flow` | visualization_msgs/MarkerArray | Gradient flow visualisation |
| `/transformed_cloud` | sensor_msgs/PointCloud2 | Input cloud transformed into the object frame |
| `/T_wc_new_pose` | geometry_msgs/PoseStamped | Estimated object pose |
| `/trajectory` | nav_msgs/Path | Object trajectory over time |

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{wu2026disflow,
  title={DisFlow: Scene Flow from Distance Field for Object Pose, Velocity Tracking and Dynamic Object Reconstruction},
  author={Wu, Lan and Sutjipto, Sheila and Wakulicz, Jennifer and Vidal-Calleja, Teresa},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2026}
}
```

For the underlying mapping framework, please also cite:

```bibtex
@article{ali2024idmp,
  title={Interactive Distance Field Mapping and Planning to Enable Human-Robot Collaboration},
  author={Ali, Usman and Wu, Lan and others},
  journal={IEEE Robotics and Automation Letters},
  year={2024}
}
```

---

## Acknowledgements

DisFlow is developed upon the IDMP framework conducted at the Robotics Institute, University of Technology Sydney.

For questions or collaborations, please contact:

**Lan Wu**  
University of Western Australia  
Email: Lan.Wu@uwa.edu.au

## Notes

This repository contains a ROS2 version of DisFlow.

The original paper was developed using ROS1. Unfortunately, after moving to a new institution, I no longer have access to the original ROS1 machine, so I decided to migrate the project to ROS2 instead of maintaining the old codebase.

At the moment, the ROS2 version has mainly been tested on the **Human Rotation** experiment presented in the paper. Some of the other experiments and utilities from the original ROS1 implementation have not yet been fully ported.

If you are interested in reproducing other results from the paper, need the original ROS1 code, rosbags, calibration files, or experiment configurations, feel free to send me an email and I will do my best to help.
