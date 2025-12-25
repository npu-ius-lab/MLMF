#ifndef _KALMAN_2D_
#define _KALMAN_2D_

#include <cstdlib>
#include <iostream>
#include <math.h>
#include <vector>
#include <set>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

class Kalman_2D
{
public:
    //构造与析构函数
    Kalman_2D();
    ~Kalman_2D();
    //滤波器初始化
    void Init(const double& Stamp, 
              const Eigen::VectorXd& X0);
    //状态预测
    void State_Predict(const double& stamp);
    //状态更新
    void State_Update(const Eigen::VectorXd& Z);
    //返回更新状态
    Eigen::VectorXd Get_X() const
    {
        return this->X_;
    }
    //滤波器预测和更新一次，并返回更新状态
    Eigen::VectorXd UpdateOnce(const double& Stamp, const Eigen::Vector4d* Z = nullptr);
    //返回预测状态
    Eigen::VectorXd Get_x() const
    {
        return this->x_;
    }

    Eigen::MatrixXd Get_P() const
    {
        return this->P_;
    }

    Eigen::MatrixXd Get_S() const
    {
        return this->S_;
    }

    int id;
    int age;
    int hit_streak;

protected:

    Eigen::MatrixXd F_;                     //状态转移矩阵 
    Eigen::MatrixXd H_;                     //观测矩阵
    Eigen::MatrixXd P_;                     //状态协方差矩阵
    Eigen::MatrixXd K_;                     //卡尔曼增益
    Eigen::MatrixXd S_;                     //观测残差协方差矩阵
    Eigen::MatrixXd R_;                     //观测噪声
    Eigen::MatrixXd Q_;                     //状态转移过程噪声

    Eigen::VectorXd X_;                     //状态向量(更新)
    Eigen::VectorXd Z_;                     //观测向量
    Eigen::VectorXd x_;                     //状态向量(预测)

    double Last_Time_Stamp_;                //上一时刻的时间戳
    double dt_;                             //时间变化量，用于计算状态转移
};

#endif