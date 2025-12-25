#ifndef _BOXSORT_
#define _BOXSORT_

#include <cstdlib>
#include <iostream>
#include <math.h>
#include <vector>
#include <set>

#include "Hungarian.h"
#include "Kalman_2D.h"

class TrackPoint
{
public:
    int id;
    Eigen::Vector4d Point; //u，v,x,y
};

class BoxSORT
{
public:
    BoxSORT();                //构造函数
    ~BoxSORT();               //析构函数
    void CalMatrix();           //计算匹配权值矩阵
    void Associate();
    void ManageTrack();
    //初始化函数
    void Init(int age, float Threshold, int minHit);
    //输入当前时刻数据                                        
    void SetData(std::vector<Eigen::Vector4d> BoxArray, double this_time);
    //跟踪                                   
    std::vector<TrackPoint> UpdateOnce(std::vector<Eigen::Vector4d> BoxArray, double this_time);  

    std::vector<TrackPoint> GetResult()
    {
        return this->TrackResult;
    };        

private:
    int max_age;                                            //目标管理最大寿命
    int trkNum;                                             //现有轨迹数量
    int detNum;                                             //检测目标数量
    float osdThreshold;                                     //匹配阈值
    int minHits;                                            //轨迹初始化步
    double this_time;                                       //当前时刻时间
    double last_time;                                       //上一时刻时间
    double dt;                                              //时间间隔
    int max_id;

    std::set<int> umTrks;
    std::set<int> umDets;
    std::vector<std::pair<int, int>> mPairs;

    //检测点集合
    std::vector<Eigen::Vector4d> detections;
    //预测点集合
    std::vector<Eigen::VectorXd> tracks;
    //欧氏距离矩阵                   
    std::vector<std::vector<double>> osdMatrix;             
    //滤波器
    std::vector<Kalman_2D> Filters;
    //存储跟踪结果
    std::vector<TrackPoint> TrackResult;

};

#endif