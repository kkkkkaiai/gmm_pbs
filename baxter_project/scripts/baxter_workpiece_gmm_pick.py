#!/usr/bin/env python
# coding=utf-8


import argparse
import time
import copy
import threading
import tf
import numpy
import baxter_left_kinematics as blk
from baxter_interface import CameraController

import rospy

from std_msgs.msg import (
    UInt16,
)
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from baxter_core_msgs.msg import EndpointState
from image_proc_msgs.msg import ImagePose

import baxter_interface

from baxter_interface import CHECK_VERSION

# 高斯模型参数
Priors_X = numpy.array([0.140423376, 0.047112043, 0.036837771, 0.00836711, 0.767259699])
Priors_Y = numpy.array([0.600735792, 0.146157602, 0.026591233, 0.206472875, 0.020042499])
Sigma_X1 = numpy.array([7707.535467, 9937.67191, 204.1946714, 1.592596122, 58845.21113])
Sigma_X2 = numpy.array([-6992.958846, -10291.83041, -201.7405654, -1.265137668, -50816.64631])
Sigma_Y1 = numpy.array([4556399.797, 890.6429231, 172.2822472, 87591.06224, 22.9766756])
Sigma_Y2 = numpy.array([-4254351.573, -656.892754, -156.7122307, -79798.58108, -17.7461887])
GMM_num = 5

# 相机参数
# 预设的相机参数值
# 深度200mm
depth = 200
# 手部相机内参
camera_cx_wrist = 307.294
camera_cy_wrist = 240.350
camera_fx_wrist = 404.940
camera_fy_wrist = 404.940

# 桌面的高度(这里为静态时抓取的高度)
static_height = -0.133
# 圆盘的高度(这里为动态物体抓取的高度)
dynamic_height = -0.073
# 设定抓取高度
pick_height = dynamic_height
# 悬停时的高度(需要等待合适的角度抓取)
hang_height = pick_height + 0.04


# 打开相机
def camera_open(_hand_camera, resolution=(640, 400), exposure=-1, gain=-1):
    hand_camera = CameraController(_hand_camera)  # left/right_hand_camera
    hand_camera.resolution = resolution
    hand_camera.exposure = exposure  # range, 0-100 auto = -1
    hand_camera.gain = gain  # range, 0-79 auto = -1
    hand_camera.white_balance_blue = -1  # range 0-4095, auto = -1
    hand_camera.white_balance_green = -1  # range 0-4095, auto = -1
    hand_camera.white_balance_red = -1  # range 0-4095, auto = -1
    # open camera
    hand_camera.open()

camera_open('left_hand_camera', exposure=15)


# 从转移矩阵中得到平移矩阵和角度
def getTfFromMatrix(matrix):
    scale, shear, angles, trans, persp = tf.transformations.decompose_matrix(matrix)
    return trans, tf.transformations.quaternion_from_euler(*angles), angles


# 监听坐标关系
def lookupTransform(tf_listener, target, source):
    tf_listener.waitForTransform(target, source, rospy.Time(), rospy.Duration(4.0))

    trans, rot = tf_listener.lookupTransform(target, source, rospy.Time())
    euler = tf.transformations.euler_from_quaternion(rot)

    source_target = tf.transformations.compose_matrix(translate=trans, angles=euler)
    return source_target


# 将四元数转化成平移矩阵
def quaternion_to_trans_matrix(x, y, z, w, trans_x, trans_y, trans_z):
    x_2, y_2, z_2 = x * x, y * y, z * z
    xy, xz, yz, wx, wy, wz = x * y, x * z, y * z, w * x, w * y, w * z
    # origin_rotation_matrix        [1 - 2 * y_2 - 2 * z_2,  2 * xy + 2 * wz,        2 * xz - 2 * wy,
    #                                2 * xy - 2 * wz,        1 - 2 * x_2 - 2 * z_2,  2 * yz + 2 * wx,
    #                                2 * xz + 2 * wy,        2 * yz - 2 * wx,        1 - 2 * x_2 - 2 * y_2]
    translation_matrix = numpy.array([1 - 2 * y_2 - 2 * z_2, 2 * xy - 2 * wz, 2 * xz + 2 * wy, trans_x,
                                      2 * xy + 2 * wz, 1 - 2 * x_2 - 2 * z_2, 2 * yz - 2 * wx, trans_y,
                                      2 * xz - 2 * wy, 2 * yz + 2 * wx, 1 - 2 * x_2 - 2 * y_2, trans_z,
                                      0, 0, 0, 1
                                      ])
    return translation_matrix.reshape((4, 4))



def get_jdagger_matrix(q):
    J = numpy.asmatrix(blk.J[6](q))
    T = numpy.asmatrix(blk.FK[6](q))
    # baxter机械臂和base坐标之间有坐标系的差异需要进行变换
    R_n_0 = T[numpy.ix_([0, 1, 2], [0, 1, 2])].transpose()
    R_c_n = numpy.matrix([[0, 1, 0],
                          [-1, 0, 0],
                          [0, 0, 1]])
    R_c_0 = R_c_n * R_n_0
    Z = numpy.asmatrix(numpy.zeros(9).reshape(3, 3))
    X1 = numpy.concatenate((R_c_0, Z), axis=1)
    X2 = numpy.concatenate((Z, R_c_0), axis=1)
    X_c_0 = numpy.concatenate((X1, X2), axis=0)
    J = X_c_0 * J
    Jt = J.transpose()
    # 求逆时可能存在行列式为0的情况,因此在对角线添加扰动来减少异常解
    # damping factor for pseudo inverse
    delta = 0.001
    I = numpy.eye(6)
    # command a desired twist and calculate needed joint angles
    Jdagger = Jt * numpy.linalg.pinv(J * Jt + delta * I)

    return Jdagger


def control_rate(error_position, ref_v):
    u_x = 0
    u_y = 0
    for i in range(GMM_num):
        u_x = u_x + Priors_X[i] * (Sigma_X2[i] / Sigma_X1[i]) * error_position[0]
        u_y = u_y + Priors_Y[i] * (Sigma_Y2[i] / Sigma_Y1[i]) * error_position[1]
    u_x = ref_v[0] - u_x
    u_y = ref_v[1] - u_y

    return u_x, u_y


# tracker类
class Tracker(object):
    def __init__(self):
        """
        Tracks a user's fist using the wrist camera.
        """
        self._pub_rate = rospy.Publisher('robot/joint_state_publish_rate', UInt16, queue_size=10)
        self._left_arm = baxter_interface.limb.Limb("left")
        self._right_arm = baxter_interface.limb.Limb("right")
        self._left_joint_names = self._left_arm.joint_names()
        self._right_joint_names = self._right_arm.joint_names()

        self.lock = threading.Lock()
        self.joint_names = ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2']

        self.joint_states = {}
        self.twist = numpy.array([0., 0., 0.])

        self.workpiece_tf = None
        self.last_wp_tf = None
        self.gripper_tf = numpy.array([0., 0., 0.])
        self.last_time = None
        self.workpiece_tf_status = False
        self.left = baxter_interface.Gripper('left', CHECK_VERSION)
        self.limb_left = baxter_interface.Limb('left')
        self.global_pause = False
        self.hang_status = False
        self.converge_status = False

        self.tf_listener = tf.TransformListener()
        self.tf_listener.waitForTransform('/left_gripper', '/base', rospy.Time(), rospy.Duration(4.0))
        gripper_temp = lookupTransform(self.tf_listener, '/left_gripper', '/base')
        trans_gripper_, _, _ = getTfFromMatrix(numpy.linalg.inv(gripper_temp))
        self.gripper_height = trans_gripper_[2]

        self.ref_v = numpy.array([0., 0.])
        self.rev_v_status = False

        self.error_pos = numpy.array([0., 0., 0.])
        self.error_pos_status = False

        # control parameters
        self._rate = 100.0  # Hz
        self.theta = 0.

        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()

        # set joint state publishing to 50Hz
        self._pub_rate.publish(self._rate)

        # control parameters
        self._rate = 100.0  # Hz

        rospy.Subscriber("/robot/joint_states", JointState, self.joint_state_callback)
        rospy.Subscriber("/workpiece_pos", ImagePose, self.image_pose_callback)
        rospy.Subscriber("/twist", Twist, self.twist_callback)

    def image_pose_callback(self, data):
        position = numpy.asarray(data.position)
        if (position[0] == -1):
            self.workpiece_tf_status = False
            return
        # x方向
        position[0] = (position[0] - camera_cx_wrist) * depth / camera_fx_wrist
        # y方向
        position[1] = (position[1] - camera_cy_wrist) * depth / camera_fy_wrist
        # 表示深度
        position[2] = depth
        # 将深度单位转换为米
        position = position / 1000
        position = numpy.append(position, 1)
        # 求出物件的空间坐标
        workpiece_marker = lookupTransform(self.tf_listener, '/left_hand_camera', '/base')
        trans_wp, rot, rot_euler = getTfFromMatrix(numpy.linalg.inv(workpiece_marker))
        translation_matrix = quaternion_to_trans_matrix(rot[0], rot[1], rot[2], rot[3],
                                                        trans_wp[0], trans_wp[1], trans_wp[2])
        base_tf = numpy.dot(translation_matrix, position)[0:3]
        self.workpiece_tf = base_tf
        # 得到当前物件的角度
        self.theta = data.orientation[0]
        self.workpiece_tf_status = True

        if self.last_wp_tf is not None:
            current_time = 0.2
            self.ref_v[0] = (self.workpiece_tf[0] - self.last_wp_tf[0]) / current_time
            self.ref_v[1] = (self.workpiece_tf[1] - self.last_wp_tf[1]) / current_time
            self.rev_v_status = True
        self.last_wp_tf = copy.copy(self.workpiece_tf)
        self.last_time = time.clock()

    def twist_callback(self, msg):
        self.twist[0] = msg.linear.x
        self.twist[1] = msg.linear.y
        self.error_pos[0] = msg.linear.x
        self.error_pos[1] = msg.linear.y
        self.error_pos_status = True

    def joint_state_callback(self, msg):
        self.joint_states = dict(zip(msg.name, msg.position))

    def _reset_control_modes(self):
        rate = rospy.Rate(self._rate)
        for _ in xrange(100):
            if rospy.is_shutdown():
                return False
            self._left_arm.exit_control_mode()
            self._right_arm.exit_control_mode()
            self._pub_rate.publish(100)  # 100Hz default joint state rate
            rate.sleep()
        return True

    def set_neutral(self):
        """
        Sets both arms back into a neutral pose.
        """
        print("Moving to neutral pose...")
        self._left_arm.move_to_neutral()
        self._right_arm.move_to_neutral()

    def clean_shutdown(self):
        print("\nExiting example...")
        # return to normal
        self._reset_control_modes()
        # self.set_neutral()
        if not self._init_state:
            print("Disabling robot...")
            self._rs.disable()
        return True

    # 根据当前的关节求出Jdagger矩阵,使用该矩阵乘上一个想要的速度值得到个个关节应达到的速度
    def track(self):
        # self.set_neutral()
        # 设定频率
        rate = rospy.Rate(50)
        self.left.command_position(100)
        # time_file = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        # time_file += '.mat'
        data = numpy.ones(6)
        data = data.reshape((1, 6))

        while not rospy.is_shutdown():
            next = False
            q = [0] * 7
            for idx, name in enumerate(self._left_joint_names):
                try:
                    q[idx] = self.joint_states[name]
                except:
                    next = True
                    break
            if next:
                continue
            # get_current_joints_state = True
            #
            # # 获取当前各个关节的角度
            # # 如果失败则跳过
            # q = [0] * 7
            # for idx, name in enumerate(self._left_joint_names):
            #     try:
            #         q[idx] = self.joint_states[name]
            #     except:
            #         get_current_joints_state = False
            #         break
            # # 如果没有得到当前关节的状态
            # if not get_current_joints_state:
            #     rospy.loginfo("current joints state error")
            #     continue

            if not self.global_pause:
                # 如果没有得到当前物块的状态

                if not self.workpiece_tf_status:
                    rospy.loginfo("workpiece tf status error")
                    continue

                # 求得Jdagger矩阵
                J = numpy.asmatrix(blk.J[6](q))
                T = numpy.asmatrix(blk.FK[6](q))
                # baxter机械臂和base坐标之间有坐标系的差异需要进行变换
                R_n_0 = T[numpy.ix_([0, 1, 2], [0, 1, 2])].transpose()
                R_c_n = numpy.matrix([[0, 1, 0],
                                      [-1, 0, 0],
                                      [0, 0, 1]])
                R_c_0 = R_c_n * R_n_0
                Z = numpy.asmatrix(numpy.zeros(9).reshape(3, 3))
                X1 = numpy.concatenate((R_c_0, Z), axis=1)
                X2 = numpy.concatenate((Z, R_c_0), axis=1)
                X_c_0 = numpy.concatenate((X1, X2), axis=0)
                J = X_c_0 * J
                Jt = J.transpose()
                # 求逆时可能存在行列式为0的情况,因此在对角线添加扰动来减少异常解
                # damping factor for pseudo inverse
                delta = 0.001
                I = numpy.eye(6)
                # command a desired twist and calculate needed joint angles
                Jdagger = Jt * numpy.linalg.pinv(J * Jt + delta * I)

                # 得到左夹爪的空间变换关系
                self.tf_listener.waitForTransform('/left_gripper', '/base', rospy.Time(), rospy.Duration(4.0))
                gripper = lookupTransform(self.tf_listener, '/left_gripper', '/base')
                trans_gripper, rot_gripper, rot_euler_gripper = getTfFromMatrix(numpy.linalg.inv(gripper))

                # 得到左手相机的空间变换关系
                self.tf_listener.waitForTransform('/left_hand_camera', '/base', rospy.Time(), rospy.Duration(4.0))
                camera = lookupTransform(self.tf_listener, '/left_hand_camera', '/base')
                trans_camera, rot_camera, rot_euler_camera = getTfFromMatrix(numpy.linalg.inv(camera))

                # 求当前z轴上的高度差(即末端高度和悬停位置)
                if self.error_pos_status:
                    dis_height = trans_gripper[2] - hang_height
                    # 下降的速度
                    ref_down_vel = 0.1
                    log_workpiece_pos = numpy.array([trans_camera[1] + self.error_pos[0], trans_camera[0] + self.error_pos[1], pick_height])

                    self.error_pos = numpy.array([trans_camera[1] - trans_gripper[1] + 0.01 + self.error_pos[0],
                    trans_camera[0] - trans_gripper[0] - 0.05 * 1 / (1 + numpy.e ** ((dis_height - 0.01) * 200)) + self.error_pos[1],
                                                  ref_down_vel*0.4])

                    # 如果到达了悬停高度,则不再进一步下降
                    if dis_height <= 0.001:
                        self.error_pos[2] = 0
                        self.hang_status = True


                else:
                    continue

                # 设定在x和y方向上的阈值,小于0.001则停止
                print(self.error_pos[0], self.error_pos[1])
                if abs(self.error_pos[0]) <= 0.01 and abs(self.error_pos[1]) <= 0.01:
                    self.converge_status = True
                    print("converge")
                elif abs(self.error_pos[0]) >= 0.05 and abs(self.error_pos[1]) <= 0.05:
                    self.converge_status = False

                # 设定默认的末端的期望速度
                gmm_twist = numpy.asmatrix([0., 0., 0., 0., 0., 0.]).transpose()
                if self.error_pos_status and self.rev_v_status:
                    # print(self.ref_v)
                    u_x, u_y = control_rate(self.error_pos, numpy.array([0, 0]))
                    # u_x, u_y = control_rate(self.error_pos, self.ref_v)
                    # u_x, u_y = self.error_pos[0],self.error_pos[1]
                else:
                    u_x = u_y = 0

                gmm_twist[0] = u_x
                gmm_twist[1] = u_y
                gmm_twist[2] = self.error_pos[2] * 0.3  # 控制夹爪下降的速度比例关系

                self.tf_listener.waitForTransform('/left_gripper', '/base', rospy.Time(), rospy.Duration(4.0))
                gripper = lookupTransform(self.tf_listener, '/left_gripper', '/base')
                trans_gripper, rot_gripper, rot_euler_gripper = getTfFromMatrix(numpy.linalg.inv(gripper))

                dis_height = (trans_gripper[2] - pick_height)  # 当前夹爪高度与设定抓取高度的误差
                ref_grip_vel = 0.15

                # gripper,
                data = numpy.append(data, numpy.array(
                    [trans_gripper[0], trans_gripper[1], trans_gripper[2],
                     log_workpiece_pos[0], log_workpiece_pos[1], log_workpiece_pos[2]]).reshape(1, 6), axis=0)

                # 如果到达了悬停的高度,则判断是不是到达了合适的抓取角度
                if self.hang_status:
                    if abs(self.theta - 90.) <= 12:
                        gmm_twist[2] = ref_grip_vel
                    else:
                        gmm_twist[2] = 0.

                # 执行抓取动作,同时准备移动停止
                if dis_height <= 0.001:
                    self.left.command_position(0)
                    self.global_pause = True

            # 如果全局停止了则表示已经抓取了物件,将物件抬起
            else:
                gmm_twist[0:3] = 0
                self.tf_listener.waitForTransform('/left_gripper', '/base', rospy.Time(), rospy.Duration(4.0))
                gripper = lookupTransform(self.tf_listener, '/left_gripper', '/base')
                trans_gripper, rot_gripper, rot_euler_gripper = getTfFromMatrix(numpy.linalg.inv(gripper))

                if trans_gripper[2] < 0.:
                    gmm_twist[2] = -0.1

            # 打印期望的速度
            # print(gmm_twist)
            if self.converge_status:
                gmm_twist[0] = 0.
                gmm_twist[1] = 0.

            # 求取关节的速度
            qDot = Jdagger * gmm_twist

            # 设定关节速度
            cmd = {}
            for idx, name in enumerate(self._left_joint_names):
                cmd[name] = qDot[idx]
            self._left_arm.set_joint_velocities(cmd)

            rate.sleep()


def main():
    """RSDK Joint Velocity Example: Tracker

    Commands joint velocities of randomly parameterized cosine waves
    to each joint. Demonstrates Joint Velocity Control Mode.
    """
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__)
    parser.parse_args(rospy.myargv()[1:])

    print("Initializing node... ")
    rospy.init_node("hand_tracker")

    tracker = Tracker()
    rospy.on_shutdown(tracker.clean_shutdown)
    tracker.track()

    print("Done.")


if __name__ == '__main__':
    main()
