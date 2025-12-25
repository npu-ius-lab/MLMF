#pragma once

#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>
#include <target_locate/TrajectoryMsg.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <tf/transform_datatypes.h>

#include <vector>
#include <cmath>

template<typename T>
T clamp_value(T v, T lo, T hi) {
    return (v < lo) ? lo : (v > hi ? hi : v);
}

template <typename T>
struct State {
    T x, y, z, yaw;

    State() : x(0), y(0), z(0), yaw(0) {}
    State(T x, T y, T z, T yaw) : x(x), y(y), z(z), yaw(yaw) {}

    static T normalize_angle(const T& angle) {
        T two_pi = T(2 * M_PI);
        T pi = T(M_PI);
        T a = angle + pi;
        a = a - floor(a / two_pi) * two_pi;
        return a - pi;
    }
};

template <typename T>
struct Control {
    T vx, vy, vz;
    Control() : vx(0), vy(0), vz(0) {}
    Control(T vx, T vy, T vz) : vx(vx), vy(vy), vz(vz) {}
};



// 传感器观测残差
struct SensorResidual {
    SensorResidual(double rx, double ry, double rz, double w)
        : rx_(rx), ry_(ry), rz_(rz), w_(w) {}

    template <typename T>
    bool operator()(const T* const x0, T* residuals) const {
        residuals[0] = T(w_) * (x0[0] - T(rx_));
        residuals[1] = T(w_) * (x0[1] - T(ry_));
        residuals[2] = T(w_) * (x0[2] - T(rz_));
        return true;
    }

    double rx_, ry_, rz_, w_;
};


// 轨迹跟踪残差
struct TrackResidual {
    TrackResidual(double tx, double ty, double tz, double w)
        : tx_(tx), ty_(ty), tz_(tz), w_(w) {}

    template <typename T>
    bool operator()(const T* const xi, T* residuals) const {
        residuals[0] = T(w_) * (xi[0] - T(tx_));
        residuals[1] = T(w_) * (xi[1] - T(ty_));
        residuals[2] = T(w_) * (xi[2] - T(tz_));
        return true;
    }

    double tx_, ty_, tz_, w_;
};


// 动力学模型残差
struct DynResidual {
    DynResidual(double dt, double w) : dt_(dt), w_(w) {}

    template <typename T>
    bool operator()(const T* const xi,
                    const T* const ui,
                    const T* const xi1,
                    T* residuals) const 
    {
        T yaw = xi[3];

        T x_pred = xi[0] + ui[0] * ceres::cos(yaw) * T(dt_)
                        - ui[1] * ceres::sin(yaw) * T(dt_);
        T y_pred = xi[1] + ui[0] * ceres::sin(yaw) * T(dt_)
                        + ui[1] * ceres::cos(yaw) * T(dt_);
        T z_pred = xi[2] + ui[2] * T(dt_);

        residuals[0] = T(w_) * (xi1[0] - x_pred);
        residuals[1] = T(w_) * (xi1[1] - y_pred);
        residuals[2] = T(w_) * (xi1[2] - z_pred);
        return true;
    }

    double dt_, w_;
};


// 输入平滑残差
struct SmoothInputResidual {
    SmoothInputResidual(double w) : w_(w) {}

    template <typename T>
    bool operator()(const T* const ui,
                    const T* const ui1,
                    T* residuals) const 
    {
        residuals[0] = T(w_) * (ui1[0] - ui[0]);
        residuals[1] = T(w_) * (ui1[1] - ui[1]);
        residuals[2] = T(w_) * (ui1[2] - ui[2]);
        return true;
    }

    double w_;
};


// 控制正则项（靠近 0）
struct InputNormResidual {
    InputNormResidual(double w) : w_(w) {}

    template <typename T>
    bool operator()(const T* const ui, T* residuals) const {
        residuals[0] = T(w_) * ui[0];
        residuals[1] = T(w_) * ui[1];
        residuals[2] = T(w_) * ui[2];
        return true;
    }

    double w_;
};



/// ===========================
///      主控类：MPCControl
/// ===========================
class MPCControl {
public:
    MPCControl(ros::NodeHandle& nh);

    void trajectoryCallback(const target_locate::TrajectoryMsg& msg);
    void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);

private:
    std::vector<double> solveMPC(double x, double y, double z, double yaw);

    ros::Publisher pub_;
    ros::Subscriber traj_sub_;
    ros::Subscriber pose_sub_;

    std::vector<geometry_msgs::Point> trajectory_points_;

    std::string drone_name_;
    double dis_x_, dis_y_;

    static constexpr int N = 5;        // MPC horizon
    static constexpr double dt = 0.1;  // 时间间隔
};
