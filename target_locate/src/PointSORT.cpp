#include "PointSORT.h"
PointSORT::PointSORT()
{
    this->max_id = 0;
}

PointSORT::~PointSORT()
{
}

//初始化SORT跟踪器（轨迹寿命， 关联阈值， 最小初始化次数）
void PointSORT::Init(int max_age, float Threshold, int minHit)
{
    this->max_age = max_age;
    this->osdThreshold = Threshold;
    this->minHits = minHit;

    this->detections.clear();
    this->Filters.clear();
    this->history_data.clear();
    this->tracks.clear();
}  

//输入当前时刻的检测目标
/*
 * @brief 1.记录识别的当前时刻this_time；清空detections,tracks存储容器,detNum记录当前识别目标个数（0）
          2.遍历detNum（0）
            detections存入该时刻每一个目标坐标（0）
            遍历Filters
              dt记录该时刻和上一次时刻差；
              根据先验估计该时刻的状态量
              获取该时刻的状态量并存储到tracks存储容器中
   @param 
          1.umDets 之前没有被识别到的新目标
 * 
 */
void PointSORT::SetData(std::vector<Eigen::Vector3d> PoseArray, double this_time)
{
    this->this_time = this_time;
    this->detections.clear();
    this->tracks.clear();
    this->detNum = PoseArray.size();
    //存入detections
    for(int i = 0; i < this->detNum; i++)
    {
        this->detections.push_back(PoseArray[i]);
    
    }
    //预测点集合
    for(int i = 0; i < this->Filters.size(); i++)
    {
        this->dt = this->this_time - this->last_time;
        this->Filters[i].Predict(this->this_time);       
        Eigen::VectorXd predict = this->Filters[i].Get_X();
        this->tracks.push_back(predict);
    }
}

//计算匹配的权值矩阵
/*
 *
        trkNum\detNum|  0  |  1  |  2  |
            0        |     |     |  *  |
            1        |  *  |     |     |
            2        |     |  *  |     |
 * 
 */
void PointSORT::CalMatrix()
{
    this->osdMatrix.clear();
    this->osdMatrix.resize(this->trkNum, std::vector<double>(this->detNum));
    
    for(int t = 0; t < this->trkNum; t++)
    {
        for(int d = 0; d < this->detNum; d++)
        {
            //使用轨迹预测点和检测点计算欧氏距离
            double xs = this->tracks[t][0] - this->detections[d][0];
            double ys = this->tracks[t][1] - this->detections[d][1];
            double zs = this->tracks[t][2] - this->detections[d][2];

            this->osdMatrix[t][d] = std::sqrt(xs * xs + ys * ys + zs * zs);
            
        }
    }
}

//数据关联
/*
 * @brief 1.清空umDets，mPairs，umTrks存储容器 ，trkNum记录当前。。。
          2.判断trkNum是否为0
            是：第一次识别到目标，将所有识别到的目标存储到umDets（0，1，2，...，len（detNum）-1）
   @param 
          1.umDets 之前没有被识别到的新目标
 * 
 */
void PointSORT::Associate()
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
        for (size_t i = 0; i < this->osdMatrix.size(); ++i) {
            // 遍历每一列
            for (size_t j = 0; j < this->osdMatrix[i].size(); ++j) {
                // 打印每个元素的值
                // ROS_INFO("osdMatrix[%zu][%zu] = %f", i, j, this->osdMatrix[i][j]);
            }
        }
        std::vector<int> assignment;
        HungarianAlgorithm HungAlgo;
        if(std::min(this->osdMatrix.size(), this->osdMatrix[0].size()) > 0)
        {
            HungAlgo.Solve(osdMatrix, assignment, false);// 索引i——跟踪点tracks；元素assignment[i]——对应检测点
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
        /*
         *
          std::set_difference比较两个向量的差集
          @param
            1.allDets是识别到的所有目标点
            2.mDets是跟跟踪点tracks匹配上的目标点
            3.umDets是跟跟踪点tracks没有匹配上的目标点但在当前时刻被识别到（新出现的目标点）
            1.allTrks是原先的所有轨迹点
            2.mTrks是跟当前时刻识别到的匹配上的轨迹点
            3.umTrks是没有在当前时刻识别到的匹配上的轨迹点但原先已经存在的轨迹点（消失的轨迹点）
         *
         */
        std::set_difference(allDets.begin(), allDets.end(), mDets.begin(), mDets.end(), std::insert_iterator<std::set<int>>(this->umDets, this->umDets.begin()));
        std::set_difference(allTrks.begin(), allTrks.end(), mTrks.begin(), mTrks.end(), std::insert_iterator<std::set<int>>(this->umTrks, this->umTrks.begin()));
        for(int i = 0; i < assignment.size(); i++)
        {
            if(assignment[i] == -1)
                continue;
            if(this->osdMatrix[i][assignment[i]] > this->osdThreshold)
            {
                double xs = this->tracks[i][0] - this->detections[assignment[i]][0];
                double ys = this->tracks[i][1] - this->detections[assignment[i]][1];
                double zs = this->tracks[i][2] - this->detections[assignment[i]][2];
                // ROS_ERROR("当前轨迹点 x=%f, y=%f, z=%f",this->tracks[i][0],this->tracks[i][1],this->tracks[i][2]);
                // ROS_ERROR("当前观测点 x=%f, y=%f, z=%f",this->detections[assignment[i]][0],this->detections[assignment[i]][1],this->detections[assignment[i]][2]);
                // ROS_ERROR("当前帧检测到的距离大于阈值：%f>%f",this->osdMatrix[i][assignment[i]],this->osdThreshold);
                // this->umDets.insert(assignment[i]);
                // this->umTrks.insert(i);
            } 
            else 
            {
                // ROS_INFO("当前轨迹点 x=%f, y=%f, z=%f",this->tracks[i][0],this->tracks[i][1],this->tracks[i][2]);
                // ROS_INFO("当前观测点 x=%f, y=%f, z=%f",this->detections[assignment[i]][0],this->detections[assignment[i]][1],this->detections[assignment[i]][2]);
                // ROS_INFO("当前帧检测到的距离小于阈值：%f<%f",this->osdMatrix[i][assignment[i]],this->osdThreshold);
                this->mPairs.push_back(std::make_pair(i, assignment[i]));
            }
        }
    }
}

//管理轨迹
/*
 * @brief 1.清空umDets，mPairs，umTrks存储容器 ，trkNum记录当前。。。
          2.为新检测到的目标初始化跟踪器，遍历umDets中每一个目标
            每一个目标用一个长度9的向量new_detection，前3维度存储x，y，z
            每一个目标新建一个滤波器
          3.清空TrackResult存储容器
            
   @param 
          1.umDets 之前没有被识别到的新目标
          2.detections 该时刻的所有目标点位置
 * 
 */
void PointSORT::ManageTrack()
{
    // 用当前匹配上的检测框和预测框加权，更新该帧的预测框
    for(int m = 0; m < this->mPairs.size(); m++)
    {
        int trkIdx = this->mPairs[m].first;
        int detIdx = this->mPairs[m].second;
        this->Filters[trkIdx].Update(this->detections[detIdx]);
    }
    // 为新检测到的目标初始化跟踪器
    for(int ud : this->umDets)
    {
        Eigen::Vector3d detection;
        Eigen::VectorXd new_detection= Eigen::VectorXd::Zero(7); //状态量 x,y,z,theta（rad）,vh,vv,w（rad/秒）
        detection = this->detections[ud];
        new_detection(0) = detection(0);
        new_detection(1) = detection(1);
        new_detection(2) = detection(2);
        EKF filter;
        filter.Init(this->this_time, new_detection);
        this->Filters.push_back(filter);
        this->history_data.emplace_back();
    }
    // 管理Trackers
    this->TrackResult.clear();
    this->ThetaResult.clear();
    this->TrajectoryResult.clear();
    for(int i = 0; i < this->Filters.size(); i++) 
    {
        if(this->Filters[i].hit_streak >= this->minHits)
        {
            if(this->Filters[i].id == -1)
            {
                //对达到初始化条件的目标赋id值
                this->max_id++;
                this->Filters[i].id = this->max_id;
                //上一个时刻指向这一时刻的角度(rad),将这个角度替换X_中的theta,
                // 获取当前状态和前一时刻状态
                Eigen::VectorXd current_X = this->Filters[i].Get_X();
                Eigen::VectorXd previous_X = this->Filters[i].Get_last_X();

                // 提取前两个元素 (x, y)
                double x_current = current_X(0);
                double y_current = current_X(1);
                double x_previous = previous_X(0);
                double y_previous = previous_X(1);

                // 计算从上一时刻到当前时刻的角度 (弧度)
                double delta_x = x_current - x_previous;
                double delta_y = y_current - y_previous;
                double angle = std::atan2(delta_y, delta_x); // 使用 atan2 确保角度范围为 [-pi, pi]

                // 更新当前状态中的 theta
                current_X(3) = angle; // theta 是向量的第 4 个元素 (索引为 3)

                // 使用 SetState 重新赋值
                this->Filters[i].SetState(current_X);
            }
            if(this->Filters[i].age <= this->max_age)
            {
                //将该帧跟踪结果存在TrackingResult
                TrackerPoint trackpoint;
                trackpoint.id = this->Filters[i].id;
                Eigen::VectorXd filter_state = this->Filters[i].Get_X();
                trackpoint.Point(0) = filter_state(0);
                trackpoint.Point(1) = filter_state(1);
                trackpoint.Point(2) = filter_state(2);
                this->TrackResult.push_back(trackpoint);
                //将该帧角度结果存在ThetaResult
                double theta = filter_state(3);
                if (filter_state(4)>0) this->ThetaResult.push_back(theta);
                else this->ThetaResult.push_back(-theta);
                //将预测的轨迹存在TrajectoryResult
                Eigen::VectorXd current_X = this->Filters[i].Get_X();
                double x = current_X(0);
                double y = current_X(1);
                double z = current_X(2);
                double dt = this->Filters[i].Get_dt();
                int num_points = Num_points;
                history_data[i].emplace_back(x, y, z);
                if (history_data[i].size() > history_len) {
                    history_data[i].erase(history_data[i].begin());
                }
                // trackpoint.history_data = history_data[i];
                // this->TrackResult.push_back(trackpoint);
                // 当历史点数量等于 history_len 时，调用预测函数
                // if (history_data[i].size() == history_len) {
                //     std::vector<Point3D> trajectory = predictTrajectory(history_data[i], dt, num_points, degree);
                //     this->TrajectoryResult.push_back(trajectory);
                // }
            }
        }
        //将多帧未检测到的滤波器给删除掉
        if(this->Filters[i].age > this->max_age)
        {
            this->Filters.erase(this->Filters.begin() + i);
            this->history_data.erase(this->history_data.begin() + i);
        }
    }
    this->last_time = this->this_time;
}

//更新函数
std::vector<TrackerPoint> PointSORT::UpdateOnce(std::vector<Eigen::Vector3d> PoseArray, double this_time)
{
    this->SetData(PoseArray, this_time);
    this->Associate();
    this->ManageTrack();

    return this->TrackResult;
}

// 多项式拟合函数
std::vector<double> PointSORT::polynomialFit(const std::vector<double>& times, const std::vector<double>& values, int degree) {
    int N = times.size();
    Eigen::MatrixXd A(N, degree + 1);
    Eigen::VectorXd b(N);

    // 构建矩阵 A 和向量 b
    for (int i = 0; i < N; ++i) {
        double t = times[i];
        for (int j = 0; j <= degree; ++j) {
            A(i, j) = std::pow(t, j); // t^j
        }
        b(i) = values[i];
    }

    // 求解最小二乘问题 (A^T * A) * coeffs = A^T * b
    Eigen::VectorXd coeffs = A.colPivHouseholderQr().solve(b);

    // 返回多项式系数
    std::vector<double> coefficients(coeffs.data(), coeffs.data() + coeffs.size());
    return coefficients;
}

// 预测轨迹点
std::vector<Point3D> PointSORT::predictTrajectory(const std::vector<Point3D>& history, double dt, int num_points, int degree) {
    int N = history.size();
    if (N < degree + 1) {
        throw std::runtime_error("Not enough points for polynomial fitting.");
    }

    // 提取时间序列（假设间隔为 dt）
    std::vector<double> times(N);
    for (int i = 0; i < N; ++i) {
        times[i] = i * dt;
    }

    // 分别提取 x, y, z 座标的历史值
    std::vector<double> x_values(N), y_values(N), z_values(N);
    for (int i = 0; i < N; ++i) {
        x_values[i] = history[i].x;
        y_values[i] = history[i].y;
        z_values[i] = history[i].z;
    }

    // 对 x, y, z 分别进行多项式拟合
    std::vector<double> x_coeffs = polynomialFit(times, x_values, degree);
    std::vector<double> y_coeffs = polynomialFit(times, y_values, degree);
    std::vector<double> z_coeffs = polynomialFit(times, z_values, degree);

    // 预测未来轨迹点
    std::vector<Point3D> trajectory;
    for (int i = 1; i <= num_points; ++i) {
        double t = N * dt + i * dt; // 预测点的时间

        // 计算拟合的多项式值
        double x = 0, y = 0, z = 0;
        for (int j = 0; j <= degree; ++j) {
            x += x_coeffs[j] * std::pow(t, j);
            y += y_coeffs[j] * std::pow(t, j);
            z += z_coeffs[j] * std::pow(t, j);
        }

        trajectory.emplace_back(x, y, z);
    }

    return trajectory;
}



// 轨迹预测，使用CTRA模拟水平运动，CA模拟垂直运动
// std::vector<Point3D> PointSORT::TrajectoryPredict(const Eigen::VectorXd& current_X, 
//                                         const Eigen::VectorXd& previous_X, 
//                                         double dt, 
//                                         int num_points) {
//     std::vector<Point3D> trajectory;

//     // 提取状态变量
//     double x = current_X(0);
//     double y = current_X(1);
//     double z = current_X(2);
//     double theta = current_X(3);
//     double v_h = current_X(4); // 水平速度
//     double v_v = current_X(5); // 垂直速度
//     double w = current_X(6);   // 角速度 (rad/s)

//     // 计算加速度（基于当前和上一时刻）
//     double prev_v_h = previous_X(4); // 上一时刻水平速度
//     double prev_v_v = previous_X(5); // 上一时刻垂直速度
//     double a_h = (v_h - prev_v_h) / dt; // 水平加速度
//     double a_v = (v_v - prev_v_v) / dt; // 垂直加速度

//     // 初始化当前位置
//     trajectory.emplace_back(x, y, z);

//     // 模拟轨迹
//     for (int i = 1; i <= num_points; ++i) {
//         // 时间更新
//         double t = i * dt;

//         // CTRA模型（水平运动：x, y, theta）
//         double new_x = x + v_h * std::cos(theta) * t + 0.5 * a_h * std::cos(theta) * t * t;
//         double new_y = y + v_h * std::sin(theta) * t + 0.5 * a_h * std::sin(theta) * t * t;
//         double new_theta = theta + w * t;

//         // CA模型（垂直运动：z）
//         double new_z = z + v_v * t + 0.5 * a_v * t * t;

//         // 添加到轨迹点
//         trajectory.emplace_back(new_x, new_y, new_z);
//     }

//     return trajectory;
// }



