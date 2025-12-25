//  Written by Guanyin Chen at Northwestern Polytechnical University in May 2023

#include "TargetLocate.h"

//像素坐标系转相机坐标系归一化平面
cv::Vec2d pixel2cam(const cv::Vec2d& p, const cv::Mat& K)
{
    cv::Vec2d Cam;
    Cam[0] = (p[0] - K.at<double>(0, 2)) / K.at<double>(0, 0);
    Cam[1] = (p[1] - K.at<double>(1, 2)) / K.at<double>(1, 1);
    
    return Cam;
}

//归一化坐标转非归一化坐标
Eigen::Vector3d normalizePoint(const cv::Mat& Point)
{
    Eigen::Vector3d Point_3d;
    Point_3d(0) = Point.at<double>(0) / Point.at<double>(3);
    Point_3d(1) = Point.at<double>(1) / Point.at<double>(3);
    Point_3d(2) = Point.at<double>(2) / Point.at<double>(3);

    return Point_3d;
}

//构造函数与析构函数
TargetLocate::TargetLocate(){}
TargetLocate::~TargetLocate(){}

//三角化计算目标坐标函数
// Eigen::Vector3d calTargetpose()
// {
    
// }


//设置相机的外参
void TargetLocate::setPose(const tf::StampedTransform& transform_1,
                           const tf::StampedTransform& transform_2)
{
    this->position_1 << transform_1.getOrigin().x(), transform_1.getOrigin().y(), transform_1.getOrigin().z();
    this->position_2 << transform_2.getOrigin().x(), transform_2.getOrigin().y(), transform_2.getOrigin().z();
    this->pose_1.x() = transform_1.getRotation().x();
    this->pose_1.y() = transform_1.getRotation().y();
    this->pose_1.z() = transform_1.getRotation().z();
    this->pose_1.w() = transform_1.getRotation().w();
    this->pose_2.x() = transform_2.getRotation().x();
    this->pose_2.y() = transform_2.getRotation().y();
    this->pose_2.z() = transform_2.getRotation().z();
    this->pose_2.w() = transform_2.getRotation().w();
}

//设置相机内参
void TargetLocate::setK(const cv::Mat& k_1,
                        const cv::Mat& k_2)
{
    this->K1 = k_1;
    this->K2 = k_2;
}

//设置检测目标的检测框中心点
void TargetLocate::setImgPoint(const cv::Vec2d& point_1,
                               const cv::Vec2d& point_2)
{
    this->img_point_1 = point_1;
    this->img_point_2 = point_2;
}

//对目标位置进行求解
void TargetLocate::target_locate()
{
    cv::Vec2d Cam1, Cam2;
    Cam1 = pixel2cam(this->img_point_1, this->K1);
    Cam2 = pixel2cam(this->img_point_2, this->K2);
    //计算外参矩阵
    Eigen::Matrix3d Rotation1, Rotation2;
    Rotation1 = this->pose_1.normalized().toRotationMatrix();
    Rotation2 = this->pose_2.normalized().toRotationMatrix();
    Eigen::MatrixXd RT1(3,4), RT2(3,4);
    RT1 << Rotation1, this->position_1;
    RT2 << Rotation2, this->position_2;
    //格式转换
    cv::Mat R1, R2;
    cv::eigen2cv(RT1, R1);
    cv::eigen2cv(RT2, R2);
    //计算空间点的坐标
    cv::Mat Point_4d;
    cv::triangulatePoints(R1, R2, Cam1, Cam2, Point_4d);
    /*
    R1,R2分别为两个相机的外参矩阵，3*4维
    Cam1,Cam2分别为目标在相机的坐标，2*1的向量
    Point_4d为计算到的目标的归一化坐标，4*1维
    */
    this->target_position = normalizePoint(Point_4d);
}

//用于获取私有属性-位置信息
Eigen::Vector3d TargetLocate::getTargetPose()
{
    return this->target_position;
}