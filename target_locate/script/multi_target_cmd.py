#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
import math

def main():
    rospy.init_node("multi_target_cmd")

    pub1 = rospy.Publisher("/target_01/cmd_vel", Twist, queue_size=10)
    pub2 = rospy.Publisher("/target_02/cmd_vel", Twist, queue_size=10)
    pub3 = rospy.Publisher("/target_03/cmd_vel", Twist, queue_size=10)

    rate = rospy.Rate(20)  # 20 Hz
    t0 = rospy.get_time()

    while not rospy.is_shutdown():
        t = rospy.get_time() - t0

        # ---------------------------
        # target_01: x+, y-, z+   
        # ---------------------------
        cmd1 = Twist()
        if t < 20.0:
            cmd1.linear.x =  0.2
            cmd1.linear.y = -0.2
            cmd1.linear.z =  0.1
        else:
            cmd1.linear.x = 0.0
            cmd1.linear.y = 0.0
            cmd1.linear.z = 0.0

        # ---------------------------
        # target_02: 逆时针圆周运动（一直执行）
        # ---------------------------
        radius = 2.0
        omega = 0.2  # 角速度 (rad/s)
        cmd2 = Twist()
        cmd2.linear.x = -radius * omega * math.sin(omega * t)
        cmd2.linear.y =  radius * omega * math.cos(omega * t)
        cmd2.linear.z =  0.0

        # ---------------------------
        # target_03: x+, y+, z−   
        # ---------------------------
        cmd3 = Twist()
        if t < 40.0:
            cmd3.linear.x =  0.2
            cmd3.linear.y =  0.2
            cmd3.linear.z = -0.1
        else:
            cmd3.linear.x = 0.0
            cmd3.linear.y = 0.0
            cmd3.linear.z = 0.0

        # 发布速度
        pub1.publish(cmd1)
        pub2.publish(cmd2)
        pub3.publish(cmd3)

        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
