#include "BoxSORT.h"

BoxSORT::BoxSORT()
{
    this->max_id = 0;
}

BoxSORT::~BoxSORT()
{
}

//初始化SORT跟踪器（轨迹寿命， 关联阈值， 最小初始化次数）
void BoxSORT::Init(int age, float Threshold, int minHit)
{
    this->max_age = age;
    this->osdThreshold = Threshold;
    this->minHits = minHit;

    this->detections.clear();
    this->Filters.clear();
    this->tracks.clear();
}  

//输入当前时刻的检测目标
void BoxSORT::SetData(std::vector<Eigen::Vector4d> BoxArray, double this_time)
{
    this->this_time = this_time;
    this->detections.clear();
    this->tracks.clear();
    this->detNum = BoxArray.size();
    //存入detections
    for(int i = 0; i < this->detNum; i++)
    {
        this->detections.push_back(BoxArray[i]);
    
    }
    //预测点集合
    for(int i = 0; i < this->Filters.size(); i++)
    {
        this->dt = this->this_time - this->last_time;
        // this->Filters[i].State_Predict(this_time);
        this->Filters[i].UpdateOnce(this->this_time);
        Eigen::VectorXd predict = this->Filters[i].Get_x();
        this->tracks.push_back(predict);
    }
}

//计算匹配的权值矩阵
void BoxSORT::CalMatrix()
{
    this->osdMatrix.clear();
    this->osdMatrix.resize(this->trkNum, std::vector<double>(this->detNum));
    
    for(int t = 0; t < this->trkNum; t++)
    {
        for(int d = 0; d < this->detNum; d++)
        {
            //使用轨迹预测点和检测点计算欧氏距离
            Eigen::Vector4d err;
            err(0) = (this->tracks[t][1] - this->tracks[t][0]) - (this->detections[d][1] - this->detections[d][0]);
            err(1) = (this->tracks[t][3] - this->tracks[t][2]) - (this->detections[d][3] - this->detections[d][2]);
            err(2) = (this->tracks[t][1] + this->tracks[t][0]) / 2.0 - (this->detections[d][1] + this->detections[d][0]) / 2.0;
            err(3) = (this->tracks[t][3] + this->tracks[t][2]) / 2.0 - (this->detections[d][3] + this->detections[d][2]) / 2.0;

            this->osdMatrix[t][d] = err.norm();
            
        }
    }
}

//数据关联
void BoxSORT::Associate()
{
    //未实现匹配的检测集合
    this->umDets.clear();
    this->mPairs.clear();
    //未实现匹配的已有目标集合
    this->umTrks.clear();
    this->trkNum = this->tracks.size();
    if(this->trkNum == 0)
    {
        for(int i = 0; i < this->detNum; i++)
            this->umDets.insert(i);
    }
    else
    {
        this->CalMatrix();
        std::vector<int> assignment;
        HungarianAlgorithm HungAlgo;
        if(std::min(this->osdMatrix.size(), this->osdMatrix[0].size()) > 0)
        {
            HungAlgo.Solve(osdMatrix, assignment, false);// 索引i——跟踪点；元素assignment[i]——对应检测点
        }
        std::set<int> allDets, allTrks, mDets, mTrks;
        for(int d = 0; d < this->detNum; d++)
            allDets.insert(d);

        for(int t = 0; t < this->trkNum; t++)
            allTrks.insert(t);
        
        if(assignment.size() > 0) 
        {
            for(int i = 0; i < this->trkNum; i++)
            {
                if(assignment[i] != -1)
                {
                    mDets.insert(assignment[i]);
                    mTrks.insert(i); 
                }
            }
        }
        if(allDets.size()>mDets.size()){
            static int frame_num = 0;
            frame_num +=1;
            if(frame_num > 3){
                std::set_difference(allDets.begin(), allDets.end(), mDets.begin(), mDets.end(), std::insert_iterator<std::set<int>>(this->umDets, this->umDets.begin()));
                frame_num = 0;
            }
        }
        std::set_difference(allDets.begin(), allDets.end(), mDets.begin(), mDets.end(), std::insert_iterator<std::set<int>>(this->umDets, this->umDets.begin()));
        std::set_difference(allTrks.begin(), allTrks.end(), mTrks.begin(), mTrks.end(), std::insert_iterator<std::set<int>>(this->umTrks, this->umTrks.begin()));

        for(int i = 0; i < assignment.size(); i++)
        {
            if(assignment[i] == -1)
                continue;
            if(this->osdMatrix[i][assignment[i]] > this->osdThreshold)
            {
                this->umDets.insert(assignment[i]);
                this->umTrks.insert(i);
            } 
            else 
            {
                this->mPairs.push_back(std::make_pair(i, assignment[i]));
            }
        }
    }
}

//管理轨迹
void BoxSORT::ManageTrack()
{
    // 用当前匹配上的检测框和预测框加权，更新该帧的预测框
    for(int m = 0; m < this->mPairs.size(); m++)
    {
        int trkIdx = this->mPairs[m].first;
        int detIdx = this->mPairs[m].second;

        // this->Filters[trkIdx].State_Update(this->detections[detIdx]);
        this->Filters[trkIdx].UpdateOnce(this->this_time, &this->detections[detIdx]);
    }
    // 为新检测到的目标初始化跟踪器
    for(int ud : this->umDets)
    {
        Eigen::Vector4d detection;
        Eigen::VectorXd new_detection= Eigen::VectorXd::Zero(6);
        detection = this->detections[ud];
        new_detection(0) = detection(0);
        new_detection(1) = detection(1);
        new_detection(2) = detection(2);
        new_detection(3) = detection(3);
        Kalman_2D filter;
        filter.Init(this->this_time, new_detection);
        filter.id = -1;
        this->Filters.push_back(filter);
    }
    // 管理Trackers
    this->TrackResult.clear();
    for(int i = 0; i < this->Filters.size(); i++) 
    {
        if(this->Filters[i].hit_streak >= this->minHits)
        {
            if(this->Filters[i].id == -1)
            {
                //对达到初始化条件的目标赋id值
                this->max_id++;
                this->Filters[i].id = this->max_id;
            }
            //将该帧跟踪结果存在TrackingResult
            if(this->Filters[i].age < this->max_age)
            {
                TrackPoint trackpoint;
                trackpoint.id = this->Filters[i].id;
                Eigen::VectorXd filter_state = this->Filters[i].Get_X();
                trackpoint.Point(0) = filter_state(0);
                trackpoint.Point(1) = filter_state(1);
                trackpoint.Point(2) = filter_state(2);
                trackpoint.Point(3) = filter_state(3);

                TrackResult.push_back(trackpoint);
            }
        }
        //将多帧未检测到的滤波器给删除掉
        if(this->Filters[i].age > this->max_age)
        {
            this->Filters.erase(this->Filters.begin() + i);
        }
    }
    this->last_time = this->this_time;
}

//更新函数
std::vector<TrackPoint> BoxSORT::UpdateOnce(std::vector<Eigen::Vector4d> BoxArray, double this_time)
{
    this->SetData(BoxArray, this_time);
    this->Associate();
    this->ManageTrack();

    return this->TrackResult;
}

