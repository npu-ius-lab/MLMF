cmake_minimum_required(VERSION 3.0.2)
project(target_locate)

add_compile_options(-std=c++14)

set(CMAKE_BUILD_TYPE Debug)
set(Ceres_DIR /usr/lib/cmake/Ceres)
set(OpenCV_DIR /usr/lib/x86_64-linux-gnu/cmake/opencv4)


find_package(catkin REQUIRED COMPONENTS 
  roscpp
  rospy
  std_msgs
  geometry_msgs
  message_filters
  tf
  cv_bridge
  darknet_ros_msgs
  message_generation
)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

add_message_files(
  FILES
  Point2D.msg
  TrajectoryMsg.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
  darknet_ros_msgs 
)

catkin_package(
  CATKIN_DEPENDS
    roscpp
    std_msgs
    geometry_msgs
    message_runtime
    darknet_ros_msgs
)


include_directories(
  include/target_locate               
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(sim_demo_node
  src/sim_demo.cpp
  src/TrajectoryPredictor.cpp
  src/Hungarian.cpp
  src/EKF.cpp                                      
  src/TargetLocate.cpp
  src/PointSORT.cpp
  src/Kalman_2D.cpp
  src/BoxSORT.cpp
)

add_executable(mpc_trajectory_follower
  src/mpc_trajectory_follower.cpp
)

add_dependencies(sim_demo_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

add_dependencies(mpc_trajectory_follower
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(sim_demo_node
  ${catkin_LIBRARIES}
  ${CERES_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

target_link_libraries(mpc_trajectory_follower
  ${catkin_LIBRARIES}
  ${CERES_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
