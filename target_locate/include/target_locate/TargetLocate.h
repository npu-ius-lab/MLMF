//  Written by Guanyin Chen at Northwestern Polytechnical University in May 2023

#ifndef _TARGETLOCATE_H_
#define _TARGETLOCATE_H_

#include <cstdlib>
#include <iostream>
#include <math.h>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <tf/transform_listener.h>

class TargetLocate
{
public:
    //设置相机的外参
    void setPose(const tf::StampedTransform& transform_1,
                 const tf::StampedTransform& transform_2);
    //设置相机内参
    void setK(const cv::Mat& k_1,
              const cv::Mat& k_2);
    //设置检测目标的检测框中心点
    void setImgPoint(const cv::Vec2d& point_1,
                     const cv::Vec2d& point_2);
    //对目标位置进行求解
    void target_locate();
    //用于获取私有属性-位置信息
    Eigen::Vector3d getTargetPose();
    //三角化计算目标坐标函数
    // Eigen::Vector3d calTargetpose(Eigen::MatrixXd RT1,
    //                               Eigen::MatrixXd RT2,
    //                               Eigen::Vector2d Cam1,
    //                               Eigen::Vector2d Cam2);
    //构造函数与析构函数
    TargetLocate();
    ~TargetLocate();

private:
    cv::Mat K1;                             //K1、K2分别为camera_matrix
    cv::Mat K2;
    Eigen::Vector3d position_1;             //position_1、position_2代表相机坐标系相对于世界坐标系的平移
    Eigen::Vector3d position_2;
    Eigen::Vector3d target_position;        //目标位置
    Eigen::Quaterniond pose_1;              //相机坐标系的姿态
    Eigen::Quaterniond pose_2;
    cv::Vec2d img_point_1;                  //目标在像素坐标的中心
    cv::Vec2d img_point_2;
};

#endif