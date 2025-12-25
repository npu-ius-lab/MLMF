#include <cstdlib>
#include <iostream>
#include <memory>
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include "EKF.h"
#include <cmath>

const float q_pos = 10;
const float q_theta = 1;
const float q_vel = 100;
const float q_w = 100;

EKF::EKF()
{
    this->age = 0; //失去观测的次数
    this->id = -1; //滤波器的id
    this->hit_streak = 1; //初始化次数
}

EKF::~EKF()
{
}

//滤波器初始化
void EKF::Init(const double& Stamp, const Eigen::VectorXd& X0)
{
    this->Last_Time_Stamp_ = Stamp;
    this->X_ = X0; //初始状态量 x,y,z,theta（rad）,vh,vv,w（rad/秒）
    //初始状态协方差
    this->P_.resize(7, 7);
    this->P_ << 100, 0, 0, 0, 0, 0, 0,
                0, 100, 0, 0, 0, 0, 0, 
                0, 0, 100, 0, 0, 0, 0,
                0, 0,  0,100, 0, 0, 0,
                0, 0,  0, 0,100, 0, 0,
                0, 0,  0, 0, 0,100, 0,
                0, 0,  0, 0, 0, 0, 100; 
    //观测矩阵(3×9)
    this->H_.resize(3, 7);  
    this->H_ << 1, 0, 0, 0, 0, 0, 0, 
                0, 1, 0, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0, 0;
    //观测噪声协方差
    this->R_.resize(3, 3);  
    this->R_ << 50, 0, 0,
                0, 50, 0,
                0, 0, 50;

}

//一步预测
void EKF::Predict(const double& Stamp)
{
    this->last_X = this->X_;
    SetCurrentTimeStamp(Stamp);
    UpdatePrediction();
    this->P_ = this->F_ * this->P_ * this->F_.transpose() + this->Q_;//预测状态协方差
    this->age++;
}

//一步更新
void EKF::Update(const Eigen::VectorXd& Z)
{
    UpdateMeasurement(Z);
    this->S_ = this->H_ * this->P_ * this->H_.transpose() + this->R_;//量测残差协方差
    this->K_ = this->P_ * this->H_.transpose() * this->S_.inverse();//计算卡尔曼增益
    this->X_ = this->X_ + this->K_ * (this->Z_ -this->H_ * this->X_);
    Eigen::MatrixXd I;
    I.setIdentity(7, 7);
    this->P_ = (I - this->K_ * this->H_) * this->P_;//更新状态协方差
    this->hit_streak++; 
    this->age = 0;
}

//设置状态协方差
void EKF::SetStateCovariance(const Eigen::MatrixXd& P) 
{
    this->P_ = P;
}

//设置状态
void EKF::SetState(const Eigen::VectorXd& X)
{
    this->X_ = X;
}

//设置当前的时间戳和时间间隔
void EKF::SetCurrentTimeStamp(const double& Stamp)
{
    this->dt_ = Stamp - this->Last_Time_Stamp_;
    this->Last_Time_Stamp_ = Stamp;
}

//测量值更新函数定义
void EKF::UpdateMeasurement(const Eigen::VectorXd& Z)
{
    this->Z_ = Z;
}

//更新预测值函数定义
void EKF::UpdatePrediction()
{
    double dt1 = this->dt_;
    double dt2 = 0.5 * dt_ * dt_;
    //状态转移矩阵（非线性运动在X_状态下的雅可比矩阵的值）
    this->F_.resize(7, 7);
    this->F_ << 1, 0, 0, -this->X_(4)*std::sin(this->X_(3))* dt1, std::cos(this->X_(3))* dt1,0, 0,
                0, 1, 0,  this->X_(4)*std::cos(this->X_(3))* dt1, std::sin(this->X_(3))* dt1,0, 0,
                0, 0, 1,  0, 0, dt1,   0, 
                0, 0, 0,  1, 0,   0, dt1,
                0, 0, 0,  0, 1,   0,   0, 
                0, 0, 0,  0, 0,   1,   0, 
                0, 0, 0,  0, 0,   0,   1;
    //过程噪声
    this->Q_.resize(7, 7);
    this->Q_ << dt2*std::cos(this->X_(3))*q_pos, 0, 0, 0, 0, 0, 0,
                0, dt2*std::sin(this->X_(3))*q_pos, 0, 0, 0, 0, 0,
                0, 0, dt2*q_pos, 0, 0, 0, 0,
                0, 0, 0, dt2*q_theta, 0, 0, 0,
                0, 0, 0, 0, dt1*q_vel, 0, 0,
                0, 0, 0, 0, 0, dt1*q_vel, 0,
                0, 0, 0, 0, 0, 0, dt1*q_w;
    UpdateState(this->X_);
}
//更新预测状态函数定义
void EKF::UpdateState(const Eigen::VectorXd& X)
{
        //CVTR模型
        // 提取当前状态变量
        double x = this->X_(0);
        double y = this->X_(1);
        double z = this->X_(2);
        double theta = this->X_(3);
        double v_h = this->X_(4);
        double v_v = this->X_(5);
        double w = this->X_(6);
        // 更新公式
        this->X_(0) = x + v_h * std::cos(theta) * this->dt_; // x
        this->X_(1) = y + v_h * std::sin(theta) * this->dt_; // y
        this->X_(2) = z + v_v* this->dt_;                   // z
        this->X_(3) = theta + w * this->dt_;                 // theta
        this->X_(4) = v_h;                            // v_h (保持不变)
        this->X_(5) = v_v;                            // v_v (保持不变)
        this->X_(6) = w;                              // w   (保持不变)
}