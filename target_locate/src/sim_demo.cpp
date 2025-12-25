#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <darknet_ros_msgs/BoundingBoxes.h>
#include <geometry_msgs/PointStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <eigen3/Eigen/Core>
#include <Eigen/Dense>

#include <target_locate/TrajectoryMsg.h>
#include <target_locate/Point2D.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "TargetLocate.h"
#include "BoxSORT.h"
#include "PointSORT.h"
#include "EKF.h"
#include "TrajectoryPredictor.h"

#define MINIMIZE_COST false // KM算法最小化代价
#define MAXIMIZE_UTIL true  // KM算法最大化收益

class MatchPair {
    public:
        TrackPoint track_points1;
        TrackPoint track_points2;

        std::vector<cv::DMatch> good_matches;
        std::vector<cv::KeyPoint> keypoints1;
        std::vector<cv::KeyPoint> keypoints2;
        Eigen::Vector3d initial_pt;

        // 默认构造函数
        MatchPair();

        // 完整构造函数：支持初始化所有成员
        MatchPair(const TrackPoint& tp1,
                  const TrackPoint& tp2,
                  const std::vector<cv::DMatch>& good_matches,
                  const std::vector<cv::KeyPoint>& kp1,
                  const std::vector<cv::KeyPoint>& kp2,
                  const Eigen::Vector3d& init_pt);
};

cv::Rect safeRect(int x, int y, int w, int h, int img_w, int img_h)
{
    x = std::max(0, x);
    y = std::max(0, y);
    w = std::min(w, img_w - x);
    h = std::min(h, img_h - y);

    if (w <= 1 || h <= 1)
        return cv::Rect(0,0,0,0);  // 返回空ROI

    return cv::Rect(x, y, w, h);
}

MatchPair::MatchPair() = default;
MatchPair::MatchPair(const TrackPoint& tp1,
    const TrackPoint& tp2,
    const std::vector<cv::DMatch>& good_matches,
    const std::vector<cv::KeyPoint>& kps1,
    const std::vector<cv::KeyPoint>& kps2,
    const Eigen::Vector3d& init_pt)
: track_points1(tp1),
track_points2(tp2),
good_matches(good_matches),
keypoints1(kps1),
keypoints2(kps2),
initial_pt(init_pt){}

void extractCameraParameters(const tf::StampedTransform& transform, const cv::Mat& K, double* camera_params) {
    // 提取平移
    double tx = transform.getOrigin().x();
    double ty = transform.getOrigin().y();
    double tz = transform.getOrigin().z();

    // 提取旋转四元数
    double qx = transform.getRotation().x();
    double qy = transform.getRotation().y();
    double qz = transform.getRotation().z();
    double qw = transform.getRotation().w();

    // 转换为轴角表示
    double quaternion[4] = {qw, qx, qy, qz};
    double axis_angle[3];
    ceres::QuaternionToAngleAxis(quaternion, axis_angle);

    // 提取相机内参
    double fx = K.at<double>(0, 0);
    double cx = K.at<double>(0, 2);
    double fy = K.at<double>(1, 1);
    double cy = K.at<double>(1, 2);

    // 填充相机参数
    camera_params[0] = axis_angle[0];
    camera_params[1] = axis_angle[1];
    camera_params[2] = axis_angle[2];
    camera_params[3] = tx;
    camera_params[4] = ty;
    camera_params[5] = tz;
    camera_params[6] = fx;
    camera_params[7] = cx;
    camera_params[8] = fy;
    camera_params[9] = cy;
}

void project_to_pixel(const cv::Mat& K, const Eigen::Vector3d& world_point,
    const std::string& camera_frame,
    tf::TransformListener& listener,
    Eigen::Vector2d& pixel_point)
{
    try 
    {
        // 构造世界系下的3D点
        geometry_msgs::PointStamped wp, cp;
        wp.header.frame_id = "world";
        wp.header.stamp = ros::Time(0);
        wp.point.x = world_point.x();
        wp.point.y = world_point.y();
        wp.point.z = world_point.z();

        // 转换到相机坐标系
        listener.waitForTransform("world", camera_frame, ros::Time(0), ros::Duration(0.1));
        listener.transformPoint(camera_frame, wp, cp);

        Eigen::Vector3d cam_point(cp.point.x, cp.point.y, cp.point.z);

        // 相机坐标系下投影到像素平面
        cv::Matx33d K_mat;
        K.convertTo(K_mat, CV_64F);  // 转成 double 类型

        Eigen::Matrix3d K_eigen;
        cv::cv2eigen(K, K_eigen);

        Eigen::Vector3d pixel_homo = K_eigen * cam_point;

        pixel_point(0) = pixel_homo(0) / pixel_homo(2);  // u
        pixel_point(1) = pixel_homo(1) / pixel_homo(2);  // v
    } 
    catch (tf::TransformException& ex) 
    {
        ROS_WARN("TF transform failed in project_to_pixel: %s", ex.what());
        pixel_point = Eigen::Vector2d(-1, -1);  // 返回非法像素点
    }
}

class ReprojectionError {
public:
    ReprojectionError(const Eigen::Vector2d& uv,
                              const cv::Mat& K,
                              double weight)
        : observed_x_(uv.x()), observed_y_(uv.y()),
          weight_(weight)
    {
        fx_ = K.at<double>(0, 0);
        fy_ = K.at<double>(1, 1);
        cx_ = K.at<double>(0, 2);
        cy_ = K.at<double>(1, 2);
    }

    template <typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    T* residuals) const 
    {
        const T* r = camera;
        const T* t = camera + 3;

        T p_cam[3];
        ceres::AngleAxisRotatePoint(r, point, p_cam);

        p_cam[0] += t[0];
        p_cam[1] += t[1];
        p_cam[2] += t[2];

        T xp = p_cam[0] / p_cam[2];
        T yp = p_cam[1] / p_cam[2];

        T u = T(fx_) * xp + T(cx_);
        T v = T(fy_) * yp + T(cy_);

        residuals[0] = T(weight_) * (u - T(observed_x_));
        residuals[1] = T(weight_) * (v - T(observed_y_));

        return true;
    }

private:
    double observed_x_, observed_y_;
    double fx_, fy_, cx_, cy_;
    double weight_;
};

class CameraPriorError {
public:
    CameraPriorError(double weight, const double* init)
        : weight_(weight)
    {
        for (int i = 0; i < 6; i++)
            init_[i] = init[i];
    }

    template <typename T>
    bool operator()(const T* const camera, T* residuals) const 
    {
        for (int i = 0; i < 6; i++)
            residuals[i] = T(weight_) * (camera[i] - T(init_[i]));

        return true;
    }

private:
    double weight_;
    double init_[6];
};

class PointPriorError {
public:
    PointPriorError(double weight, const Eigen::Vector3d& init)
        : weight_(weight), init_(init)
    {}

    template<typename T>
    bool operator()(const T* const point, T* residuals) const
    {
        residuals[0] = T(weight_) * (point[0] - T(init_.x()));
        residuals[1] = T(weight_) * (point[1] - T(init_.y()));
        residuals[2] = T(weight_) * (point[2] - T(init_.z()));
        return true;
    }

private:
    double weight_;
    Eigen::Vector3d init_;
};

class LocateNode {
public:
    LocateNode(ros::NodeHandle& nh) : 
        tf_listener(tf_buffer),
        predictor(history_len, Num_points)
    {
        // 获取相机内参
        nh.getParam("camera_matrix_01", cam_matrix_01_);
        nh.getParam("camera_matrix_02", cam_matrix_02_);
        nh.getParam("/camera_link_1", camera_link_1_);
        nh.getParam("/camera_link_2", camera_link_2_);
        K1_ = cv::Mat(3, 3, CV_64F, cam_matrix_01_.data()).clone();
        K2_ = cv::Mat(3, 3, CV_64F, cam_matrix_02_.data()).clone();

        // 初始化跟踪器
        point_tracker.Init(10, 1.5, 50);
        box_tracker_1.Init(10, 200, 10);
        box_tracker_2.Init(10, 200, 10);

        // 初始化订阅器
        sub_box1_.subscribe(nh, "/detect_msg_1", 1);
        sub_box2_.subscribe(nh, "/detect_msg_2", 1);
        sub_img1_.subscribe(nh, "/image_1", 1);
        sub_img2_.subscribe(nh, "/image_2", 1);

        sync_.reset(new Sync(SyncPolicy(20), sub_box1_, sub_box2_, sub_img1_, sub_img2_));
        sync_->registerCallback(boost::bind(&LocateNode::callback, this, _1, _2, _3, _4));

        // 初始化发布器
        marker_pub = nh.advertise<visualization_msgs::MarkerArray>("tracking/targets", 10);
        traj_pub = nh.advertise<target_locate::TrajectoryMsg>("trajectory_point", 10);
        ray_pub = nh.advertise<visualization_msgs::MarkerArray>("camera_rays", 10);
        predictor.setPublisher(traj_pub); 
        std::cout<<"init LocateNode"<<std::endl;
        // pub_target_ = nh.advertise<geometry_msgs::PointStamped>("/target_position", 1);
    }

private:
    //ros
    ros::NodeHandle nh;
    // 相机参数
    std::vector<double> cam_matrix_01_, cam_matrix_02_;
    std::string camera_link_1_, camera_link_2_;
    cv::Mat K1_, K2_;
    // 时间参数
    bool set_stamp = true;     // 用于判断是否为第一帧
    double Stamp0;           // 第一帧的时间戳记录

    // TF
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener;
    tf::TransformListener listener;

    // 订阅器
    message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes> sub_box1_, sub_box2_;
    message_filters::Subscriber<sensor_msgs::Image> sub_img1_, sub_img2_;

    // 发布器
    ros::Publisher marker_pub;
    ros::Publisher traj_pub;
    ros::Publisher ray_pub;

    //定位器
    TargetLocate locate;

    //跟踪器
    BoxSORT box_tracker_1;
    BoxSORT box_tracker_2;
    PointSORT point_tracker;

    // 同步器
    typedef message_filters::sync_policies::ApproximateTime<
        darknet_ros_msgs::BoundingBoxes,
        darknet_ros_msgs::BoundingBoxes,
        sensor_msgs::Image,
        sensor_msgs::Image> SyncPolicy;
    typedef message_filters::Synchronizer<SyncPolicy> Sync;
    std::shared_ptr<Sync> sync_;

    // 轨迹预测器
    TrajectoryPredictor predictor;

    //跟踪目标id
    int target_id;

    //回调
    void callback(const darknet_ros_msgs::BoundingBoxes::ConstPtr& detect_1,
                  const darknet_ros_msgs::BoundingBoxes::ConstPtr& detect_2,
                  const sensor_msgs::Image::ConstPtr& image_1,
                  const sensor_msgs::Image::ConstPtr& image_2);
};

void LocateNode::callback(const darknet_ros_msgs::BoundingBoxes::ConstPtr& detect_1,
    const darknet_ros_msgs::BoundingBoxes::ConstPtr& detect_2,
    const sensor_msgs::Image::ConstPtr& image_1,
    const sensor_msgs::Image::ConstPtr& image_2) 
{
    //***********记录时间戳***********//
    // ros消息时间戳转换成double类型的时间
    double Stamp_1 = detect_1->header.stamp.toSec();
    double Stamp_2 = detect_2->header.stamp.toSec();
    if (set_stamp)
    {
        set_stamp = false;
        // 设置第一帧时间戳作为初始时间戳
        Stamp0 = std::min(Stamp_1, Stamp_2);
    }
    // 当前时刻的时间间隔，用于滤波器和数据同步
    Stamp_2 = Stamp_2 - Stamp0;
    Stamp_1 = Stamp_1 - Stamp0;

    ROS_INFO(
        "\n-----------------------------------\n"
        "Time = %.3f\n"
        "-----------------------------------\n",
        Stamp_1
    );

    //***********检查目标***********//
    if (detect_1->bounding_boxes.empty() || detect_2->bounding_boxes.empty()) return;

    // 检测目标个数
    int detNum_1 = detect_1->bounding_boxes.size(); 
    int detNum_2 = detect_2->bounding_boxes.size(); 
    ROS_INFO(
        "detNum_1: %d\n"
        "detNum_2: %d\n",
        detNum_1,detNum_2
    );
    //检测框中心点，长宽
    std::vector<Eigen::Vector4d> BoxArray_1, BoxArray_2;
    for (const auto& box : detect_1->bounding_boxes)
    {
        BoxArray_1.emplace_back(
            (box.xmin + box.xmax) / 2.0,
            (box.ymin + box.ymax) / 2.0,
            box.xmax - box.xmin,
            box.ymax - box.ymin
        );
    }
    for (const auto& box : detect_2->bounding_boxes)
    {
        BoxArray_2.emplace_back(
            (box.xmin + box.xmax) / 2.0,
            (box.ymin + box.ymax) / 2.0,
            box.xmax - box.xmin,
            box.ymax - box.ymin
        );
    }
    //***********SORT跟踪器***********//
    std::vector<TrackPoint> BoxSet_1 = box_tracker_1.UpdateOnce(BoxArray_1, Stamp_1);
    ROS_INFO("rmtt_01 跟踪结果：");
    for (int i = 0; i < BoxSet_1.size(); i++)
    {
        ROS_INFO("id=%d x=%.2f y=%.2f w=%.2f h=%.2f",
                        BoxSet_1[i].id,
                        BoxSet_1[i].Point[0],
                        BoxSet_1[i].Point[1],
                        BoxSet_1[i].Point[2],
                        BoxSet_1[i].Point[3]);
    }
    std::vector<TrackPoint> BoxSet_2 = box_tracker_2.UpdateOnce(BoxArray_2, Stamp_2);
    ROS_INFO("rmtt_02 跟踪结果：");
    for (int i = 0; i < BoxSet_2.size(); i++)
    {
        ROS_INFO("id=%d x=%.2f y=%.2f w=%.2f h=%.2f",
                        BoxSet_2[i].id,
                        BoxSet_2[i].Point[0],
                        BoxSet_2[i].Point[1],
                        BoxSet_2[i].Point[2],
                        BoxSet_2[i].Point[3]);
    }
    if ( BoxSet_1.size() == 0 || BoxSet_2.size() == 0 ) return;
    //////////////////////////////////////////////
    // 可视化：从相机发出的射线（Ray Visualization）
    //////////////////////////////////////////////

    visualization_msgs::MarkerArray ray_array;
    int ray_id = 0;

    // 射线长度
    double ray_length = 20.0;

    // 获取相机位姿（world系）
    tf::StampedTransform cam1_tf, cam2_tf;
    listener.lookupTransform("world", camera_link_1_, ros::Time(0), cam1_tf);
    listener.lookupTransform("world", camera_link_2_, ros::Time(0), cam2_tf);

    // 相机 1：像素射线
    for (size_t i = 0; i < BoxSet_1.size(); i++)
    {
        int u = BoxSet_1[i].Point[0];
        int v = BoxSet_1[i].Point[1];

        // 归一化方向
        double x = (u - K1_.at<double>(0,2)) / K1_.at<double>(0,0);
        double y = (v - K1_.at<double>(1,2)) / K1_.at<double>(1,1);

        Eigen::Vector3d dir_cam(x, y, 1.0);
        dir_cam.normalize();

        // 转到世界系
        tf::Vector3 origin1 = cam1_tf.getOrigin();
        tf::Matrix3x3 R1 = cam1_tf.getBasis();
        tf::Vector3 dir_w1 = R1 * tf::Vector3(dir_cam.x(), dir_cam.y(), dir_cam.z());

        // 生成射线（Marker）
        visualization_msgs::Marker ray;
        ray.header.frame_id = "world";
        ray.header.stamp = ros::Time::now();
        ray.ns = "camera1_rays";
        ray.id = ray_id++;
        ray.type = visualization_msgs::Marker::ARROW;

        ray.scale.x = 0.01; // 箭头粗细
        ray.scale.y = 0.1;
        ray.scale.z = 0.1;

        ray.color.r = 0.0f;
        ray.color.g = 1.0f;
        ray.color.b = 0.0f;
        ray.color.a = 1.0f;

        geometry_msgs::Point p_start, p_end;
        p_start.x = origin1.x();
        p_start.y = origin1.y();
        p_start.z = origin1.z();

        p_end.x = origin1.x() + ray_length * dir_w1.x();
        p_end.y = origin1.y() + ray_length * dir_w1.y();
        p_end.z = origin1.z() + ray_length * dir_w1.z();

        ray.points.push_back(p_start);
        ray.points.push_back(p_end);

        ray_array.markers.push_back(ray);
    }


    // 相机 2：像素射线
    for (size_t j = 0; j < BoxSet_2.size(); j++)
    {
        int u = BoxSet_2[j].Point[0];
        int v = BoxSet_2[j].Point[1];

        double x = (u - K2_.at<double>(0,2)) / K2_.at<double>(0,0);
        double y = (v - K2_.at<double>(1,2)) / K2_.at<double>(1,1);

        Eigen::Vector3d dir_cam(x, y, 1.0);
        dir_cam.normalize();

        tf::Vector3 origin2 = cam2_tf.getOrigin();
        tf::Matrix3x3 R2 = cam2_tf.getBasis();
        tf::Vector3 dir_w2 = R2 * tf::Vector3(dir_cam.x(), dir_cam.y(), dir_cam.z());

        visualization_msgs::Marker ray;
        ray.header.frame_id = "world";
        ray.header.stamp = ros::Time::now();
        ray.ns = "camera2_rays";
        ray.id = ray_id++;
        ray.type = visualization_msgs::Marker::ARROW;

        ray.scale.x = 0.01;
        ray.scale.y = 0.1;
        ray.scale.z = 0.1;

        ray.color.r = 1.0f;
        ray.color.g = 0.0f;
        ray.color.b = 0.0f;
        ray.color.a = 1.0f;

        geometry_msgs::Point p_start, p_end;
        p_start.x = origin2.x();
        p_start.y = origin2.y();
        p_start.z = origin2.z();

        p_end.x = origin2.x() + ray_length * dir_w2.x();
        p_end.y = origin2.y() + ray_length * dir_w2.y();
        p_end.z = origin2.z() + ray_length * dir_w2.z();

        ray.points.push_back(p_start);
        ray.points.push_back(p_end);

        ray_array.markers.push_back(ray);
    }

    // 发布所有射线
    ray_pub.publish(ray_array);

    //***********获取所有匹配下的初始定位***********//
    // 构建分配矩阵
    tf::StampedTransform transform1;
    tf::StampedTransform transform2;
    // 设置两无人机相机内参数
    locate.setK(K1_, K2_);
    // 设置两无人机相机外参数
    listener.lookupTransform(camera_link_1_, "world", ros::Time(0), transform1);
    listener.lookupTransform(camera_link_2_, "world", ros::Time(0), transform2);
    locate.setPose(transform1, transform2);
    //建立匹配对应关系
    std::vector<std::vector<double>> likelihood_Matrix(BoxSet_1.size(), std::vector<double>(BoxSet_2.size(), -1000));
    std::vector<std::vector<Eigen::Vector3d>> target_position_Matrix(BoxSet_1.size(), std::vector<Eigen::Vector3d>(BoxSet_2.size()));

    for (int i = 0; i < BoxSet_1.size(); ++i) 
    {
        for (int j = 0; j < BoxSet_2.size(); ++j) 
        {
            cv::Vec2d p1 = {BoxSet_1[i].Point(0), BoxSet_1[i].Point(1) };
            cv::Vec2d p2 = {BoxSet_2[j].Point(0), BoxSet_2[j].Point(1) };

            locate.setImgPoint(p1, p2);
            locate.target_locate();
            Eigen::Vector3d target_position = locate.getTargetPose();
            ROS_INFO(
                "%d 配对 %d : (%.3f, %.3f, %.3f)",
                BoxSet_1[i].id,
                BoxSet_2[j].id,
                target_position(0),
                target_position(1),
                target_position(2)
            );
            // 重投影误差计算
            Eigen::Vector2d err_1, err_2;
            Eigen::Vector2d pixel_point_1, pixel_point_2;
            project_to_pixel(K1_, target_position, camera_link_1_, listener, pixel_point_1);
            project_to_pixel(K2_, target_position, camera_link_2_, listener, pixel_point_2);

            Eigen::Vector2d vp1 = {BoxSet_1[i].Point(0), BoxSet_1[i].Point(1) };
            Eigen::Vector2d vp2 = {BoxSet_2[j].Point(0), BoxSet_2[j].Point(1) };
            
            err_1 = pixel_point_1 - vp1;
            err_2 = pixel_point_2 - vp2;

            double mod = err_1.norm() + err_2.norm();
            double likelihood = -(200 * (std::atan(mod / 20.0 - 2) + M_PI / 2.0)) + 600;
            if (likelihood < 400) likelihood = -1000;
            likelihood_Matrix[i][j] = likelihood;
            std::cout<<"likelihood:"<<likelihood<<std::endl;
            target_position_Matrix[i][j] = target_position;
        }
    }

    // 匈牙利匹配
    HungarianAlgorithm hungAlgo;
    std::vector<int> assignment;
    double cost = hungAlgo.Solve(likelihood_Matrix, assignment, MAXIMIZE_UTIL);
    for (int i = 0; i < assignment.size(); i++)
    {
        int j = assignment[i];
        if (j == -1) continue;

        if (likelihood_Matrix[i][j] < 0)
        {
            assignment[i] = -1;  // 无效匹配，拒绝
        }
    }

    //存储点
    std::vector<MatchPair> matchpairs;
    for (int i = 0; i < assignment.size(); ++i) {
        if (assignment[i] == -1) continue;
        Eigen::Vector3d pt = target_position_Matrix[i][assignment[i]];
        ROS_INFO("Matched point: x = %f, y = %f, z = %f", pt.x(), pt.y(), pt.z());
        
        //特征提取
        cv::Mat img1 = cv_bridge::toCvShare(image_1, "bgr8")->image;
        int cx1 = BoxSet_1[i].Point[0];
        int cy1 = BoxSet_1[i].Point[1];
        int w1  = BoxSet_1[i].Point[2];
        int h1  = BoxSet_1[i].Point[3];

        int xmin1 = cx1 - w1/2;
        int ymin1 = cy1 - h1/2;

        cv::Rect rect1 = safeRect(
            xmin1, ymin1, w1, h1,
            img1.cols, img1.rows
        );
        cv::Mat roi1 = img1(rect1).clone();
        
        std::vector<cv::KeyPoint> keypoints1;
        cv::Ptr<cv::ORB> detector1 = cv::ORB::create();
        detector1->setMaxFeatures(200);  // 设置要检测的特征点数量
        detector1->setNLevels(2);  // 设置尺度金字塔的层数
        detector1->setEdgeThreshold(2);  // 设置边缘阈值
        detector1->setFirstLevel(0);  // 设置尺度金字塔的起始层级
        detector1->setWTA_K(2);  // 设置描述符计算时的 WTA_K 值
        detector1->setScoreType(cv::ORB::FAST_SCORE);  // 设置角点检测的分数类型
        detector1->detect(roi1, keypoints1);

        if (keypoints1.empty()) continue;
        for (auto &keypoint : keypoints1)
        {
            keypoint.pt += cv::Point2f(rect1.x, rect1.y);
        }
        // 计算描述符
        cv::Mat descriptors1;
        detector1->compute(img1, keypoints1, descriptors1);

        cv::Mat img2 = cv_bridge::toCvShare(image_2, "bgr8")->image;
        int cx2 = BoxSet_2[assignment[i]].Point[0];
        int cy2 = BoxSet_2[assignment[i]].Point[1];
        int w2  = BoxSet_2[assignment[i]].Point[2];
        int h2  = BoxSet_2[assignment[i]].Point[3];

        int xmin2 = cx2 - w2/2;
        int ymin2 = cy2 - h2/2;

        cv::Rect rect2 = safeRect(
            xmin2, ymin2, w2, h2,
            img2.cols, img2.rows
        );
        cv::Mat roi2 = img2(rect2).clone();
        
        std::vector<cv::KeyPoint> keypoints2;
        cv::Ptr<cv::ORB> detector2 = cv::ORB::create();
        detector2->setMaxFeatures(200);  // 设置要检测的特征点数量
        detector2->setNLevels(2);  // 设置尺度金字塔的层数
        detector2->setEdgeThreshold(2);  // 设置边缘阈值
        detector2->setFirstLevel(0);  // 设置尺度金字塔的起始层级
        detector2->setWTA_K(2);  // 设置描述符计算时的 WTA_K 值
        detector2->setScoreType(cv::ORB::FAST_SCORE);  // 设置角点检测的分数类型
        detector2->detect(roi2, keypoints2);

        if (keypoints2.empty()) continue;
        for (auto &keypoint : keypoints2)
        {
            keypoint.pt += cv::Point2f(rect2.x, rect2.y);
        }
        // 计算描述符
        cv::Mat descriptors2;
        detector2->compute(img2, keypoints2, descriptors2);

        cv::BFMatcher matcher(cv::NORM_HAMMING, true);
        std::vector<cv::DMatch> matches;
        matcher.match(descriptors1, descriptors2, matches);

        // 距离筛选
        double avg_dist = 0;
        for (const auto& m : matches) avg_dist += m.distance;
        avg_dist /= matches.size();

        std::vector<cv::DMatch> good_matches;
        for (const auto& m : matches) {
            if (m.distance < avg_dist * 10) good_matches.push_back(m);
        }

        cv::Mat match_result;
        cv::drawMatches(img1, keypoints1, img2, keypoints2, good_matches, match_result);
        std::string window_name = "ORB Matching: " + std::to_string(i) + " -> " + std::to_string(assignment[i]);
        ROS_INFO(
            "匹配结果 %d -> %d : 选取特征点数量 = %lu",
            BoxSet_1[i].id,
            BoxSet_2[assignment[i]].id,
            good_matches.size()
        );
        cv::imshow(window_name, match_result);
        cv::waitKey(1);

        MatchPair pair(BoxSet_1[i], BoxSet_2[assignment[i]], good_matches, keypoints1, keypoints2, pt);
        matchpairs.push_back(pair);
    }
    
    /////////////////////// 联合优化 BA /////////////////////////

    int num_cameras = 2;

    // 相机参数数组（旋转+平移）
    double camera1[6];
    double camera2[6];

    double w_reproj = 1.0;     // 重投影误差权重
    double w_cam_prior = 150.0; // 相机外参先验权重
    double w_pt_prior  = 0.3;  // 目标点先验权重

    // 从 transform + K 中提取 angle-axis + translation
    extractCameraParameters(transform1, K1_, camera1);
    extractCameraParameters(transform2, K2_, camera2);

    ROS_INFO("Initial Camera 1: r=[%.3f %.3f %.3f], t=[%.3f %.3f %.3f]",
            camera1[0], camera1[1], camera1[2],
            camera1[3], camera1[4], camera1[5]);

    ROS_INFO("Initial Camera 2: r=[%.3f %.3f %.3f], t=[%.3f %.3f %.3f]",
            camera2[0], camera2[1], camera2[2],
            camera2[3], camera2[4], camera2[5]);
    
    ceres::Problem problem;

    // ============ 1. 创建所有 3D 点（目标点）参数块 ============
    std::vector<Eigen::Vector3d> points3d;
    points3d.reserve(matchpairs.size());

    for (size_t i = 0; i < matchpairs.size(); ++i)
    {
        points3d.push_back(matchpairs[i].initial_pt);
        problem.AddParameterBlock(points3d[i].data(), 3);
    }

    double camera1_initial[6];
    double camera2_initial[6];
    for (int i = 0; i < 6; i++) {
        camera1_initial[i] = camera1[i];
        camera2_initial[i] = camera2[i];
    }
    std::vector<Eigen::Vector3d> points_initial = points3d;

    //加点参数
    for (size_t i = 0; i < points3d.size(); i++)
        problem.AddParameterBlock(points3d[i].data(), 3);

    //加相机参数
    problem.AddParameterBlock(camera1, 6);
    problem.AddParameterBlock(camera2, 6);

    //重投影误差
    for (size_t i = 0; i < matchpairs.size(); ++i)
    {
        const MatchPair& pair = matchpairs[i];
        Eigen::Vector3d& pt = points3d[i];

        for (const auto& m : pair.good_matches)
        {
            // cam1
            Eigen::Vector2d uv1(pair.keypoints1[m.queryIdx].pt.x,
                                pair.keypoints1[m.queryIdx].pt.y);

            ceres::CostFunction* cost1 =
                new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
                    new ReprojectionError(uv1, K1_, w_reproj));

            problem.AddResidualBlock(cost1, nullptr, camera1, pt.data());

            // cam2
            Eigen::Vector2d uv2(pair.keypoints2[m.trainIdx].pt.x,
                                pair.keypoints2[m.trainIdx].pt.y);

            ceres::CostFunction* cost2 =
                new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
                    new ReprojectionError(uv2, K2_, w_reproj));

            problem.AddResidualBlock(cost2, nullptr, camera2, pt.data());
        }
    }

    //相机外参先验
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<CameraPriorError, 6, 6>(
            new CameraPriorError(w_cam_prior, camera1_initial)),
        nullptr,
        camera1
    );

    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<CameraPriorError, 6, 6>(
            new CameraPriorError(w_cam_prior, camera2_initial)),
        nullptr,
        camera2
    );

    //点先验
    for (size_t i = 0; i < points3d.size(); ++i)
    {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<PointPriorError, 3, 3>(
                new PointPriorError(w_pt_prior, points_initial[i])),
            nullptr,
            points3d[i].data()
        );
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    ROS_INFO("BA Optimization Summary:\n%s", summary.BriefReport().c_str());

    std::vector<Eigen::Vector3d> target_poses;
    target_poses.reserve(points3d.size());

    for (size_t i = 0; i < points3d.size(); ++i)
    {
        ROS_INFO("Optimized %lu: x=%.3f y=%.3f z=%.3f",
                i, points3d[i].x(), points3d[i].y(), points3d[i].z());

        target_poses.push_back(points3d[i]);
    }

    /////////////////////// 联合优化结束 /////////////////////////

    //EKF跟踪器
    std::vector<TrackerPoint> point_trackers = point_tracker.UpdateOnce(target_poses, Stamp_1);
    ROS_INFO("跟踪目标点：%lu 个", target_poses.size());
    for (int i = 0; i < point_trackers.size(); i++)
    {
        ROS_INFO("滤波后的点 %d:      x = %f, y = %f, z = %f", point_trackers[i].id, 
            point_trackers[i].Point(0), point_trackers[i].Point(1), point_trackers[i].Point(2));
    }
    //发布位置
    visualization_msgs::MarkerArray marray;
    for (size_t i = 0; i < point_trackers.size(); ++i)
    {
        TrackerPoint p = point_trackers[i];
        if (p.id<4){
            // 添加 Text 显示 ID
            visualization_msgs::Marker text_marker;
            text_marker.header.frame_id = "world";
            text_marker.header.stamp = detect_1->header.stamp;
            text_marker.ns = "id_text";
            text_marker.id = p.id;
            text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            text_marker.pose.position.x = p.Point(0);
            text_marker.pose.position.y = p.Point(1);
            text_marker.pose.position.z = p.Point(2) + 0.2; // Text 偏高显示
            text_marker.pose.orientation.w = 1.0;
            text_marker.scale.z = 0.3; 
            text_marker.color.r = 0.0f;
            text_marker.color.g = 1.0f;
            text_marker.color.b = 0.8f;
            text_marker.color.a = 1.0f;
            text_marker.text = "id=" + std::to_string(p.id);
            text_marker.lifetime = ros::Duration(0.15);
            marray.markers.push_back(text_marker);
            
            // 添加 Sphere 显示当前位置
            visualization_msgs::Marker sphere_marker;
            sphere_marker.header.frame_id = "world";
            sphere_marker.header.stamp = detect_1->header.stamp;
            sphere_marker.ns = "current_position";
            sphere_marker.id = i;
            sphere_marker.type = visualization_msgs::Marker::SPHERE;
            sphere_marker.pose.position.x = p.Point(0);
            sphere_marker.pose.position.y = p.Point(1);
            sphere_marker.pose.position.z = p.Point(2);
            sphere_marker.pose.orientation.w = 1.0;
            sphere_marker.scale.x = 0.3; 
            sphere_marker.scale.y = 0.3;
            sphere_marker.scale.z = 0.3;
            sphere_marker.color.r = 0.0f;
            sphere_marker.color.g = 1.0f;
            sphere_marker.color.b = 0.0f;
            sphere_marker.color.a = 1.0f;
            sphere_marker.lifetime = ros::Duration(0.15);
            marray.markers.push_back(sphere_marker);
        }
    }

    // 发布 MarkerArray
    marker_pub.publish(marray);

    /////////////////////////记录轨迹点////////////////////////////////
    predictor.run(
        point_tracker.Get_TrackerPoint(),
        target_poses,
        point_tracker.Get_history_data()
    );
}


int main(int argc, char** argv) {
    ros::init(argc, argv, "demo_node");
    ros::NodeHandle nh;

    LocateNode node(nh);

    ros::spin();
    return 0;
}