#ifndef _EKF_H
#define _EKF_H

#include <cstdlib>
#include <iostream>
#include <memory>
#include <cmath>
#include <vector>
#include <algorithm>
#include <eigen3/Eigen/Dense>

class EKF
{
public:
    //构造与析构函数
    EKF();
    ~EKF();
    //滤波器初始化
    void Init(const double& Stamp, 
              const Eigen::VectorXd& X0);
    //状态预测
    void Predict(const double& stamp);
    //状态更新
    void Update(const Eigen::VectorXd& Z);
    //设置状态协方差
    void SetStateCovariance(const Eigen::MatrixXd& P);
    //设置状态
    void SetState(const Eigen::VectorXd& X);
    //设置当前的时间戳和时间间隔
    void SetCurrentTimeStamp(const double& Stamp);
    //测量值更新函数定义
    void UpdateMeasurement(const Eigen::VectorXd& Z);
    //更新预测值函数定义
    void UpdatePrediction();
    //更新预测状态函数定义
    void UpdateState(const Eigen::VectorXd& X);
    //返回更新状态
    Eigen::VectorXd Get_X() const
    {
        return this->X_;
    }
    Eigen::VectorXd Get_last_X() const
    {
        return this->last_X;
    }
    Eigen::MatrixXd Get_P() const
    {
        return this->P_;
    }

    Eigen::MatrixXd Get_S() const
    {
        return this->S_;
    }
    double Get_dt() const
    {
        return this->dt_;
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

    Eigen::VectorXd last_X;                 //上一时刻的状态向量
    Eigen::VectorXd X_;                     //状态向量
    Eigen::VectorXd Z_;                     //观测向量

    double Last_Time_Stamp_;                //上一时刻的时间戳
    double dt_;                             //时间变化量，用于计算状态转移
};

#endif