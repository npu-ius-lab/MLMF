#!/usr/bin/python3
import rospy
from geometry_msgs.msg import Twist, PoseStamped
import time
class UAVAltitudeController:
    def __init__(self):
        rospy.init_node("uav_altitude_controller")

        # UAV 名称与目标高度（单位：米）
        self.uav_list = {
            'obs_01': 2.0,
            'obs_02': 2.0,
            'target_01': 2.5,
            'target_02': 2.0,
            'target_03': 1.5
        }

        self.pose_data = {}  # 存储每个 UAV 的当前 pose
        self.publishers = {}  # 存储每个 UAV 的 cmd_vel 发布器

        # 订阅 pose，并创建对应的 cmd_vel 发布器
        for uav, target_z in self.uav_list.items():
            rospy.Subscriber(f"/{uav}/ground_truth_to_tf/pose", PoseStamped, self.pose_callback, callback_args=uav)
            self.publishers[uav] = rospy.Publisher(f"/{uav}/cmd_vel", Twist, queue_size=10)

        self.rate = rospy.Rate(10)  # 10Hz 控制频率
        self.control_loop()

    def pose_callback(self, msg, uav_name):
        self.pose_data[uav_name] = msg.pose

    def control_loop(self):
        while not rospy.is_shutdown():
            for uav, target_z in self.uav_list.items():
                if uav in self.pose_data:
                    current_z = self.pose_data[uav].position.z
                    error_z = target_z - current_z

                    # 简单的P控制器
                    vz = 0.8 * error_z

                    # 限制最大速度
                    vz = max(min(vz, 0.5), -0.5)

                    cmd = Twist()
                    cmd.linear.z = vz
                    self.publishers[uav].publish(cmd)

            self.rate.sleep()

if __name__ == "__main__":
    time.sleep(5)
    try:
        UAVAltitudeController()
    except rospy.ROSInterruptException:
        pass
