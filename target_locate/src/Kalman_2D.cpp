
#include "Kalman_2D.h"

Kalman_2D::Kalman_2D()
{
    this->age = 0;
    this->hit_streak = 1;
}

Kalman_2D::~Kalman_2D()
{
}

//滤波器初始化
void Kalman_2D::Init(const double& Stamp, 
                     const Eigen::VectorXd& X0)
{
    this->Last_Time_Stamp_ = Stamp;
    this->X_ = X0; //(x,y,w,h,dx,dy)
    //初始状态协方差
    this->P_.setIdentity(6, 6);
    this->P_ = 100 * this->P_;
    //观测矩阵(4×6)
    this->H_.resize(4, 6);  
    this->H_ << 1, 0, 0, 0, 0, 0, 
                0, 1, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0,
                0, 0, 0, 1, 0, 0;
                
    //观测噪声协方差
    this->R_.resize(4, 4);  
    this->R_ << 1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 2, 0,
                0, 0, 0, 2;
    
}

//滤波器预测步
void Kalman_2D::State_Predict(const double& Stamp)
{
    this->dt_ = Stamp - this->Last_Time_Stamp_;
    this->Last_Time_Stamp_ = Stamp;
    double dt1 = this->dt_;
    //状态转移矩阵
    this->F_.resize(6, 6);
    this->F_ << 1, 0,  0,   0,    dt1,      0, 
                0, 1,  0,   0,      0,    dt1, 
                0, 0,  1,   0,      0,      0, 
                0, 0,  0,   1,      0,      0,       
                0, 0,  0,   0,      1,      0,       
                0, 0,  0,   0,      0,      1,      

    //
    this->Q_.resize(6, 6);
    this->Q_ << 10*dt1, 0,       0,       0,       0,       0,
                0, 10*dt1,       0,       0,       0,       0,     
                0,      0,   5*dt1,       0,       0,       0,    
                0,      0,       0,   5*dt1,       0,       0,      
                0,      0,       0,       0,   5*dt1,       0,       
                0,      0,       0,       0,       0,    5*dt1;
    this->x_ = this->F_ * this->X_;
    this->P_ = this->F_ * this->P_ * this->F_.transpose() + this->F_ * this->Q_ * this->F_.transpose();
}

//滤波器更新步
void Kalman_2D::State_Update(const Eigen::VectorXd& Measure)
{
    Eigen::VectorXd v = Measure - this->H_ * this->x_;
    this->S_ = this->H_ * this->P_ * this->H_.transpose() + this->R_;
    this->K_ = this->P_ * this->H_.transpose() * this->S_.inverse();
    this->X_ = this->x_ + this->K_ * v;
    this->P_ = this->P_ - this->K_ * this->H_ * this->P_;
}

Eigen::VectorXd Kalman_2D::UpdateOnce(const double& Stamp, const Eigen::Vector4d* Z)
{
    if(Z == nullptr)
    {
        //无观测数据，仅预测，age+1
        Kalman_2D::State_Predict(Stamp);
        this->X_ = this->x_;

        this->age++;
    }
    else
    {
        //有观测数据，状态更新，age = 0
        Kalman_2D::State_Predict(Stamp);
        Kalman_2D::State_Update(*Z);

        this->hit_streak++;
        this->age = 0;
    }
    

    return Kalman_2D::Get_X();
}