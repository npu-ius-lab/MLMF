#pragma once

#include <vector>
#include <cmath>
#include <limits>
#include <ros/ros.h>
#include <Eigen/Dense>
#include "target_locate/TrajectoryMsg.h"
#include "PointSORT.h"   

struct TrajectoryInfo {
    double speed = 0.0;
    double radius = 0.0;
    std::vector<Point3D> closest_trajectory;
    std::vector<Point3D> Predict_Trajectory;
};

class TrajectoryPredictor {
public:
    TrajectoryPredictor(int _history_len, int _num_predict);
    void setPublisher(ros::Publisher pub);

    void run(const std::vector<TrackerPoint>& trackerpoints,
             const std::vector<Eigen::Vector3d>& target_points,
             const std::vector<std::vector<Point3D>>& history_trajs);

private:
    // 内部计算函数
    TrajectoryInfo findFollowTrajectory(const std::vector<std::vector<Point3D>>& history_trajs,
                                        const Point3D& p);

    double computeDistance(const Point3D& a, const Eigen::Vector3d& b);
    double computeDistance(const Point3D& a, const Point3D& b);
    double computeAngle(const Point3D& center, const Point3D& p);
    void calculateCircle(const Point3D& p1, const Point3D& p2, const Point3D& p3,
                         Point3D& center, double& radius);
    Point3D getNextPoint(const Point3D& center, double radius,
                         double angle, double angular_vel, double dt);

    void processTrajectory(const std::vector<Point3D>& traj,
                           target_locate::TrajectoryMsg& msg);

private:
    int history_len;
    int num_predict;

    ros::Publisher traj_pub_;
    bool has_pub_ = false;
};
