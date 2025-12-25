# MLMF

Code for the paper "Multi-target Association and Localization with Distributed Drone Following: A Factor Graph Approach".

## **Abstract**

Vision-based multi-drone multi-object tracking technology enables autonomous target situational awareness for unmanned aerial systems. Distributed observer drones dynamically estimate the spatio-temporal states of multiple targets through collaborative sensor fusion, enabling simultaneous localization and persistent following of the target of interest in cluttered airspaces. The challenge lies in distinguishing targets in different drones’ views and keeping the target of interest within the field of view. This paper proposes a factor graph method for joint multi-target association and localization with distributed drone following. Sensor measurements and control constraints are integrated into a probabilistic factor graph to solve the bundle adjustment and model predictive control, respectively. Both simulation and real-world experiments prove the effectiveness and robustness of our proposed approach.

## **Installation**

The code has been tested on ​**Ubuntu 20.04 + ROS Noetic**​. GPU is recommended for running YOLOv5-based detection.
### **1. Install YOLOv5**

Follow the original YOLOv5 installation instructions (conda environment recommended):

```bash
conda create -n yolov5 python=3.8
conda activate yolov5
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

### **2. Install ROS YOLO messages**

```bash
sudo apt-get install ros-noetic-darknet-ros-msgs
```

### **3. Clone our repository**

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/npu-ius-lab/MLMF
cd ..
```

### **4. Install required C++ libraries**

Our project depends on:

* Eigen3
* OpenCV 4
* Ceres Solver
* ROS message generation

Install them via apt:

```bash
sudo apt-get install libeigen3-dev
sudo apt-get install libopencv-dev
sudo apt-get install libceres-dev
sudo apt-get install libc6-dev
```


### **5. Build the workspace**

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

## **Run our Code**



### **1. Launch the Gazebo simulation**

```bash
roslaunch hector_quadrotor_demo demo.launch
```

### **2. Launch YOLOv5 detection**


```bash
roslaunch yolov5_ros gazebo_tt.launch
```

### **3. Start the multi-drone localization node**

```bash
roslaunch target_locate sim_locate_demo.launch
```


### **4. Select the target ID to follow**

Example: choose target 2:

```bash
rosparam set target_id 2
```
### **5. Two observer drones track the selected target**

```bash
roslaunch target_locate two_drone_mpc_tracking.launch
```

### **6. Multi-target multi-drone coordinated control**


```bash
roslaunch target_locate multi_target_control.launch
```


## **Citation**

If you use this code or our paper, please cite:

```
@inproceedings{inproceedings,
author = {Ye, Kaixiao and Shao, Weiyu and Zheng, Yuhang and Fang, Bohui and Yang, Tao},
year = {2025},
month = {10},
pages = {18856-18863},
title = {Multi-target Association and Localization with Distributed Drone Following: A Factor Graph Approach},
doi = {10.1109/IROS60139.2025.11247561}
}
```


## **Acknowledgements**

We sincerely thank the open-source community for providing valuable tools and datasets that greatly supported this project.
In particular, we acknowledge the following repositories:

* **YOLOv5** — for real-time object detection and perception:
  [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* **DbMOT**  — for multi-platform multi-target tracking frameworks:
  [https://github.com/npu-ius-lab/DbMOT](https://github.com/npu-ius-lab/DbMOT)
* **NPU RoboCourse Sim** — for providing UAV simulation environments and ROS-based teaching frameworks:
  [https://github.com/npu-ius-lab/npurobocourse\_sim](https://github.com/npu-ius-lab/npurobocourse_sim)

We thank all contributors and maintainers of these projects for their dedication to open-source research and robotics education.
