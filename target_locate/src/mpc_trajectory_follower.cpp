#include "mpc_trajectory_follower.h"

/// ===========================
///    构造函数
/// ===========================
MPCControl::MPCControl(ros::NodeHandle& nh)
{
    ros::NodeHandle pnh("~");

    pnh.param("drone_name", drone_name_, std::string("rmtt_01"));
    dis_x_ = pnh.param("dis_x", 0.0);
    dis_y_ = pnh.param("dis_y", 0.0);

    pub_ = nh.advertise<geometry_msgs::Twist>("/" + drone_name_ + "/cmd_vel", 10);
    traj_sub_ = nh.subscribe("/trajectory_point", 10, &MPCControl::trajectoryCallback, this);
    pose_sub_ = nh.subscribe("/" + drone_name_ + "/ground_truth_to_tf/pose", 10, &MPCControl::poseCallback, this);
}


/// ===========================
///  轨迹订阅
/// ===========================
void MPCControl::trajectoryCallback(const target_locate::TrajectoryMsg& msg)
{
    trajectory_points_.clear();
    for (auto& p : msg.trajectory_points)
        trajectory_points_.push_back(p);
}


/// ===========================
///    位姿订阅 → 触发MPC
/// ===========================
void MPCControl::poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    double x = msg->pose.position.x;
    double y = msg->pose.position.y;
    double z = msg->pose.position.z;
    double yaw = tf::getYaw(msg->pose.orientation);

    if (trajectory_points_.empty()) return;

    std::vector<double> u = solveMPC(x, y, z, yaw);

    /// 发布控制
    geometry_msgs::Twist cmd;
    cmd.linear.x = clamp_value(u[0], -0.4, 0.4);
    cmd.linear.y = clamp_value(u[1], -0.4, 0.4);
    cmd.linear.z = clamp_value(u[2], -0.3, 0.3);
    pub_.publish(cmd);
}


/// ===========================
///     核心：MPC 求解
/// ===========================
std::vector<double> MPCControl::solveMPC(double x0, double y0, double z0, double yaw0)
{
    /// 待优化变量
    double X[N+1][4];     // x,y,z,yaw
    double U[N][3];       // vx,vy,vz

    /// 初始化状态
    for (int i=0; i<=N; i++) {
        X[i][0] = x0;
        X[i][1] = y0;
        X[i][2] = z0;
        X[i][3] = yaw0;
    }

    /// 初始化控制
    for (int i=0; i<N; i++) {
        U[i][0] = 0;
        U[i][1] = 0;
        U[i][2] = 0;
    }

    ceres::Problem problem;

    /// 1. 传感器残差（约束 X0）
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<SensorResidual, 3, 4>(
            new SensorResidual(x0, y0, z0, 40.0)),
        nullptr,
        X[0]
    );

    /// 2. 轨迹跟踪残差（每一时刻）
    for (int i=0; i<N && i < trajectory_points_.size(); i++) {
        auto& t = trajectory_points_[i];
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<TrackResidual, 3, 4>(
                new TrackResidual(t.x + dis_x_, t.y + dis_y_, t.z, 20.0)),
            nullptr,
            X[i]
        );
    }

    /// 3. 动力学残差（X[i+1] = f(X[i], U[i])）
    for (int i=0; i<N; i++) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<DynResidual, 3, 4, 3, 4>(
                new DynResidual(dt, 50.0)),
            nullptr,
            X[i], U[i], X[i+1]
        );
    }

    /// 4. 控制平滑项
    for (int i=0; i<N-1; i++) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<SmoothInputResidual, 3, 3, 3>(
                new SmoothInputResidual(20.0)),
            nullptr,
            U[i], U[i+1]
        );
    }

    /// 5. 控制正则项
    for (int i=0; i<N; i++) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<InputNormResidual, 3, 3>(
                new InputNormResidual(5.0)),
            nullptr,
            U[i]
        );
    }

    /// 求解器
    ceres::Solver::Options opt;
    opt.max_num_iterations = 50;
    opt.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    ceres::Solve(opt, &problem, &summary);
    // std::cout << summary.BriefReport() << std::endl;

    /// 返回第一个输入（真正用于控制的）
    return {U[0][0], U[0][1], U[0][2]};
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "mpc_trajectory_follower");
    ros::NodeHandle nh;

    MPCControl mpc(nh);

    ros::spin();
    return 0;
}
