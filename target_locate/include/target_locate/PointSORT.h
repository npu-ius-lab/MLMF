#ifndef _POINTSORT_
#define _POINTSORT_

#include <cstdlib>
#include <iostream>
#include <math.h>
#include <vector>
#include <set>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <ros/ros.h>
#include "Hungarian.h"
#include "EKF.h"

const int Num_points = 30;//预测轨迹点数量
const int history_len = 30;//历史轨迹点数量
const int degree = 2; //多项式拟合阶数
class TrackerPoint
{
public:
    int id;
    Eigen::Vector3d Point;
};

// 定义一个点结构体表示三维空间中的轨迹点
struct Point3D {
    double x;
    double y;
    double z;
    Point3D() : x(0.0), y(0.0), z(0.0) {}
    Point3D(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
};

class PointSORT
{
public:
    PointSORT();                //构造函数
    ~PointSORT();               //析构函数
    void CalMatrix();           //计算匹配权值矩阵
    void Associate();
    void ManageTrack();
    //初始化函数
    void Init(int max_age, float Threshold, int minHit);
    //输入当前时刻数据                                        
    void SetData(std::vector<Eigen::Vector3d> PoseArray, double this_time);
    //跟踪                                   
    std::vector<TrackerPoint> UpdateOnce(std::vector<Eigen::Vector3d> PoseArray, double this_time);  

    std::vector<std::vector<Point3D>> Get_Trajectory()
    {
        return this->TrajectoryResult;
    }
    std::vector<float> Get_Theta()
    {
        return this->ThetaResult;
    }
    double Get_dt()
    {
        return this->dt;
    }

    std::vector<TrackerPoint> GetResult()
    {
        return this->TrackResult;
    };        
    std::vector<std::vector<Point3D>> Get_history_data()
    {
        return this->history_data;
    };
    std::vector<TrackerPoint> Get_TrackerPoint()
    {
        return this->TrackResult;
    };
    std::vector<double> polynomialFit(const std::vector<double>& times, const std::vector<double>& values, int degree);
    std::vector<Point3D> predictTrajectory(const std::vector<Point3D>& history, double dt, int num_points, int degree);

private:
    int max_age;                                            //目标管理最大寿命
    int trkNum;                                             //现有轨迹数量
    int detNum;                                             //检测目标数量
    float osdThreshold;                                     //匹配阈值
    // float osdThreshold_false;
    int minHits;                                            //轨迹初始化步
    double this_time;                                       //当前时刻时间
    double last_time;                                       //上一时刻时间
    double dt;                                              //时间间隔
    int max_id;

    std::set<int> umTrks;
    std::set<int> umDets;
    std::vector<std::pair<int, int>> mPairs;

    //检测点集合
    std::vector<Eigen::VectorXd> detections;
    //预测点集合
    std::vector<Eigen::VectorXd> tracks;
    //欧氏距离矩阵                   
    std::vector<std::vector<double>> osdMatrix;             
    //滤波器
    std::vector<EKF> Filters;
    //历史轨迹存储器
    std::vector<std::vector<Point3D>> history_data;
    //存储跟踪结果
    std::vector<TrackerPoint> TrackResult;
    //存储跟踪角度结果
    std::vector<float> ThetaResult;
    //存储预测轨迹点结果
    std::vector<std::vector<Point3D>> TrajectoryResult;
};

#endif