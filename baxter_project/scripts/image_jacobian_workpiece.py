#!/usr/bin/env python
import rospy
import numpy as np
import math as m
import copy
from tld_msgs.msg import BoundingBox
from geometry_msgs.msg import Twist
from image_proc_msgs.msg import ImagePose

half_width = 320
half_height = 200
u = 0  # target location in pixel
w = 0
confidence = 0

# f = 2000   # focal_length in pixel
f = 405.8
alpha = 1  # tuning parameter
depth = 200
camera_cx_wrist = 307.294
camera_cy_wrist = 240.350
camera_fx_wrist = 404.940
camera_fy_wrist = 404.940
call_status = False


def callback(data):
    global call_status
    if data.position[0] == -1 and data.position[1] == -1:
        call_status = False
        return
    global u
    global w
    global confidence
    u = (data.position[0] - camera_cx_wrist) * depth / camera_fx_wrist
    w = (data.position[1] - camera_cy_wrist) * depth / camera_fy_wrist
    confidence = data.position[2]
    call_status = True


def calculate_command():
    global u
    global w
    global alpha
    denominator = 1.0 / (m.pow(f, 2) + m.pow(u, 2) + m.pow(w, 2))
    angular_command = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # print(denominator)
    if confidence == 0:
        pass
    else:
        # print("w: ",w, "u: ",u)
        angular_command[0] = (u) * (depth + camera_fx_wrist) / camera_fx_wrist / 1000
        angular_command[1] = (w) * (depth + camera_fy_wrist) / camera_fy_wrist / 1000
        # print(angular_command[0], angular_command[1])
        angular_command_ = denominator * np.array([-alpha * w * f, alpha * u * f, 2 * alpha * u * w])
        angular_command[3] = angular_command_[0]
        angular_command[4] = angular_command_[1]
        angular_command[5] = angular_command_[2]

    return angular_command


def listener():
    rospy.init_node('image_jacobian', anonymous=True)
    rospy.Subscriber("/workpiece_pos", ImagePose, callback)
    pub = rospy.Publisher("/twist", Twist, queue_size=1)

    rate = rospy.Rate(30)  # 10hz
    global call_status
    while not rospy.is_shutdown():
        if call_status:
            angular_command = calculate_command()
            # print angular_command
            msg = Twist()
            msg.linear.x = -angular_command[0]
            msg.linear.y = -angular_command[1]
            msg.angular.x = -angular_command[3]
            msg.angular.y = -angular_command[4]
            msg.angular.z = -angular_command[5]
            pub.publish(msg)
        rate.sleep()

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    listener()
