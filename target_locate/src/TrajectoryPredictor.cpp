#include "TrajectoryPredictor.h"

TrajectoryPredictor::TrajectoryPredictor(int _history_len, int _num_predict)
    : history_len(_history_len), num_predict(_num_predict)
{
}

void TrajectoryPredictor::setPublisher(ros::Publisher pub)
{
    traj_pub_ = pub;
    has_pub_ = true;
}

// ==============================
//        计算工具函数
// ==============================

double TrajectoryPredictor::computeDistance(const Point3D& a, const Eigen::Vector3d& b) {
    return (Eigen::Vector3d(a.x, a.y, a.z) - b).norm();
}

double TrajectoryPredictor::computeDistance(const Point3D& a, const Point3D& b) {
    return std::sqrt((a.x - b.x)*(a.x - b.x) +
                     (a.y - b.y)*(a.y - b.y) +
                     (a.z - b.z)*(a.z - b.z));
}

double TrajectoryPredictor::computeAngle(const Point3D& c, const Point3D& p) {
    return std::atan2(p.y - c.y, p.x - c.x);
}

void TrajectoryPredictor::calculateCircle(
    const Point3D& p1, const Point3D& p2, const Point3D& p3,
    Point3D& center, double& radius)
{
    double d = 2 * (p1.x*(p2.y - p3.y) + p2.x*(p3.y - p1.y) + p3.x*(p1.y - p2.y));
    if (fabs(d) < 1e-6) {    // 三点共线
        center = Point3D(0,0,0);
        radius = -1;
        return;
    }

    double ux = ((p1.x*p1.x + p1.y*p1.y)*(p2.y - p3.y) +
                 (p2.x*p2.x + p2.y*p2.y)*(p3.y - p1.y) +
                 (p3.x*p3.x + p3.y*p3.y)*(p1.y - p2.y)) / d;

    double uy = ((p1.x*p1.x + p1.y*p1.y)*(p3.x - p2.x) +
                 (p2.x*p2.x + p2.y*p2.y)*(p1.x - p3.x) +
                 (p3.x*p3.x + p3.y*p3.y)*(p2.x - p1.x)) / d;

    center = Point3D(ux, uy, (p1.z+p2.z+p3.z)/3);
    radius = computeDistance(center, p1);
}

Point3D TrajectoryPredictor::getNextPoint(const Point3D& center, double radius,
                                          double angle, double angular_vel, double dt)
{
    double new_angle = angle + angular_vel * dt;
    return Point3D(center.x + radius * std::cos(new_angle),
                   center.y + radius * std::sin(new_angle),
                   center.z);
}

// ==============================
//         轨迹预测函数
// ==============================

TrajectoryInfo TrajectoryPredictor::findFollowTrajectory(
    const std::vector<std::vector<Point3D>>& history_trajs, 
    const Point3D& p)
{
    TrajectoryInfo info;
    double min_dist = std::numeric_limits<double>::max();
    int best_idx = -1;

    // 1. 找当前点最近的历史轨迹
    for (size_t i = 0; i < history_trajs.size(); ++i) {
        if (history_trajs[i].size() != history_len) continue;

        double d = computeDistance(history_trajs[i].back(), p);
        if (d < min_dist) {
            min_dist = d;
            best_idx = i;
        }
    }

    if (best_idx == -1) return info;

    auto traj = history_trajs[best_idx];

    // 2. 判断是否静止
    double dist_move = computeDistance(traj.front(), traj.back());
    if (dist_move < 0.2) {
        Point3D avg;
        for (auto& t : traj) {
            avg.x += t.x; avg.y += t.y; avg.z += t.z;
        }
        avg.x/=traj.size(); avg.y/=traj.size(); avg.z/=traj.size();

        for (int i=0;i<num_predict;i++) {
            info.Predict_Trajectory.emplace_back(
                avg.x+0.01*i, avg.y+0.01*i, avg.z+0.005*i
            );
        }
        return info;
    }

    // 3. 三点拟合圆
    Point3D center;
    double radius;
    int idx1 = history_len - num_predict;
    int idx2 = history_len - num_predict/2;
    calculateCircle(traj[idx1], traj[idx2], p, center, radius);

    if (radius < 0.1) return info;

    double angle_now = computeAngle(center, p);
    double angle_prev = computeAngle(center, traj[idx1]);
    double dt = 0.1;
    double angular_vel = (angle_now - angle_prev) / (num_predict * dt);

    for (int i = 0; i < num_predict; ++i) {
        info.Predict_Trajectory.push_back(
            getNextPoint(center, radius, angle_now, angular_vel, dt)
        );
        angle_now += angular_vel * dt;
    }

    info.radius = radius;
    info.speed = radius * angular_vel;
    info.closest_trajectory = traj;

    return info;
}

// ==============================
//        发布消息
// ==============================

void TrajectoryPredictor::processTrajectory(const std::vector<Point3D>& traj,
                                            target_locate::TrajectoryMsg& msg)
{
    int idx = 0;
    for (size_t i=0; i<traj.size() && idx<8; i+=3) {
        geometry_msgs::Point pt;
        pt.x = traj[i].x; pt.y = traj[i].y; pt.z = traj[i].z;
        msg.trajectory_points[idx++] = pt;
    }
}

// ==============================
//       主流程 run()
// ==============================

void TrajectoryPredictor::run(
    const std::vector<TrackerPoint>& trackerpoints,
    const std::vector<Eigen::Vector3d>& target_points,
    const std::vector<std::vector<Point3D>>& history_trajs)
{
    if (!has_pub_) {
        ROS_ERROR("TrajectoryPredictor: publisher 未设置！");
        return;
    }

    int target_id;
    if (!ros::param::get("/target_id", target_id)) {
        ROS_WARN("target_id 未设置");
        return;
    }

    // -------------------------------
    // 找 trackerpoint
    // -------------------------------
    const TrackerPoint* target = nullptr;
    for (auto& p : trackerpoints) {
        if (p.id == target_id) {
            target = &p;
            break;
        }
    }
    if (!target) return;

    // -------------------------------
    // 找最近 target 点
    // -------------------------------
    Point3D closest;
    double min_dist = std::numeric_limits<double>::infinity();

    for (auto& v : target_points) {
        Point3D pt(v(0), v(1), v(2));
        double d = computeDistance(pt, target->Point);
        if (d < min_dist) {
            min_dist = d;
            closest = pt;
        }
    }

    // -------------------------------
    // 生成预测点
    // -------------------------------
    TrajectoryInfo info = findFollowTrajectory(history_trajs, closest);

    if (info.Predict_Trajectory.empty()) return;

    // -------------------------------
    // 发布
    // -------------------------------
    target_locate::TrajectoryMsg msg;
    processTrajectory(info.Predict_Trajectory, msg);
    traj_pub_.publish(msg);
}
