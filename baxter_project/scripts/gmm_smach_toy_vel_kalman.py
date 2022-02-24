#!/usr/bin/env python
# coding=utf-8

import argparse
import time
import copy
import threading
import tf
import numpy
import numpy as np
import os
import baxter_left_kinematics as blk
from baxter_interface import CameraController
import smach
import smach_ros
import rospy

from std_msgs.msg import (
    UInt16,
)
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from baxter_core_msgs.msg import EndpointState
# from image_proc_msgs.msg import ImagePose

import baxter_interface
from baxter_interface import CHECK_VERSION

# 高斯模型参数
# sim param
Priors_X = numpy.array([0.140423376, 0.047112043, 0.036837771, 0.00836711, 0.767259699])
Priors_Y = numpy.array([0.600735792, 0.146157602, 0.026591233, 0.206472875, 0.020042499])
Sigma_X1 = numpy.array([7707.535467, 9937.67191, 204.1946714, 1.592596122, 58845.21113])
Sigma_X2 = numpy.array([-6992.958846, -10291.83041, -201.7405654, -1.265137668, -50816.64631])
Sigma_Y1 = numpy.array([4556399.797, 890.6429231, 172.2822472, 87591.06224, 22.9766756])
Sigma_Y2 = numpy.array([-4254351.573, -656.892754, -156.7122307, -79798.58108, -17.7461887])

# real param 2
# Priors_X = numpy.array([0.0573513, 0.659728, 0.0568025, 0.0473145, 0.1788037])
# Priors_Y = numpy.array([0.4979475, 0.2689866, 0.0030122, 0.027888, 0.202172])
# Sigma_X1 = numpy.array([6.64, 0.086914665, 893., 13.26703312, 955.])
# Sigma_X2 = numpy.array([-6.47, -0.099, -484., -15.18964978, -606.])
# Sigma_Y1 = numpy.array([5.717040663, 24.98822829, 9330., 8.526799355, 20.4])
# Sigma_Y2 = numpy.array([-1.308130165, -15.42068391, -2390., -31.18523148, -8.46])

# real param 3
# Priors_X = numpy.array([0.0573513, 0.0473145, 0.0568025, 0.0473145, 0.1788037])
# Priors_Y = numpy.array([0.4979475, 0.2689866, 0.0030122, 0.027888, 0.202172])
# Sigma_X1 = numpy.array([6.64, 0.086914665, 893., 13.26703312, 955.])
# Sigma_X2 = numpy.array([-6.47, -0.099, -484., -15.18964978, -606.])
# Sigma_Y1 = numpy.array([5.717040663, 24.98822829, 9330., 8.526799355, 20.4])
# Sigma_Y2 = numpy.array([-1.308130165, -15.42068391, -2390., -31.18523148, -8.46])

GMM_num = 5

# 相机参数
# 预设的相机参数值

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
pick_height = static_height
# 悬停时的高度(需要等待合适的角度抓取)
hang_height = pick_height + 0.1
converge = 0.01

rospy.init_node('gmm_smach')


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


# camera_open('left_hand_camera', exposure=15)


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
    translation_matrix = numpy.array([1 - 2 * y_2 - 2 * z_2, 2 * xy - 2 * wz, 2 * xz + 2 * wy, trans_x,
                                      2 * xy + 2 * wz, 1 - 2 * x_2 - 2 * z_2, 2 * yz - 2 * wx, trans_y,
                                      2 * xz - 2 * wy, 2 * yz + 2 * wx, 1 - 2 * x_2 - 2 * y_2, trans_z,
                                      0, 0, 0, 1
                                      ])
    return translation_matrix.reshape((4, 4))


def control_rate(error_position, ref_v):
    # 控制率
    # x/y方向上的
    u_x = 0
    u_y = 0
    for i in range(GMM_num):
        u_x = u_x + Priors_X[i] * (Sigma_X2[i] / Sigma_X1[i]) * error_position[0]
        u_y = u_y + Priors_Y[i] * (Sigma_Y2[i] / Sigma_Y1[i]) * error_position[1]
    u_x = ref_v[0] - u_x
    u_y = ref_v[1] - u_y

    return u_x, u_y


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


# 卡尔曼滤波用于跟踪物体速度
class KalmanFilterVel:
    def __init__(self):
        dt = 0.06
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        self.H = np.eye(4)

        self.Q = np.eye(4) * 0.1

        self.R = np.eye(4)

        self.B = None
        self.P = np.eye(4)

        self.initial_state = None
        self.P_posterior = np.array(self.P)

    def set_initial_state(self, initial_state):
        self.initial_state = initial_state
        self.X_posterior = np.array(self.initial_state)
        self.Z = np.array(self.initial_state)

    def update(self, position, status):
        if self.initial_state is None:
            print("State is not initial.")
            return None
        last_posterior = self.X_posterior[0:2]

        if status:
            dx = position[0] - self.X_posterior[0]
            dy = position[1] - self.X_posterior[1]

            self.Z[0:2] = np.array([position])
            self.Z[2::] = np.array([dx, dy])

        if status:
            # -----进行先验估计-----------------
            X_prior = np.dot(self.A, self.X_posterior)
            # -----计算状态估计协方差矩阵P--------
            P_prior_1 = np.dot(self.A, self.P_posterior)
            P_prior = np.dot(P_prior_1, self.A.T) + self.Q
            # ------计算卡尔曼增益---------------------
            k1 = np.dot(P_prior, self.H.T)
            k2 = np.dot(np.dot(self.H, P_prior), self.H.T) + self.R
            K = np.dot(k1, np.linalg.inv(k2))
            # --------------后验估计------------
            X_posterior_1 = self.Z - np.dot(self.H, X_prior)
            self.X_posterior = X_prior + np.dot(K, X_posterior_1)
            vel_posterior = self.X_posterior[2::]
            # ---------更新状态估计协方差矩阵P-----
            P_posterior_1 = np.eye(4) - np.dot(K, self.H)
            self.P_posterior = np.dot(P_posterior_1, P_prior)
        else:
            self.X_posterior = np.dot(self.A, self.X_posterior)
            vel_posterior = self.X_posterior[2::]

        return vel_posterior


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
        self._left_arm.set_joint_position_speed(0.8)

        self.lock = threading.Lock()
        self.joint_names = ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2']

        self.joint_states = {}
        self.twist = numpy.array([0., 0., 0.])

        self.workpiece_tf = None
        self.last_wp_tf = None
        self.gripper_tf = numpy.array([0., 0., 0.])

        self.last_time = None
        self.workpiece_pos = None  # 用于保存通过像素求出的差值
        self.workpiece_tf_status = False

        self.left = baxter_interface.Gripper('left', CHECK_VERSION)
        self.limb_left = baxter_interface.Limb('left')
        self.left.command_position(100)  # 打开夹爪

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
        self.rate = rospy.Rate(50)  # Hz
        self.theta = 0.

        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()

        # kalman_filter parameter
        self.kal_vel = KalmanFilterVel()  # instance the KalmanFilterVel class
        self.last_pos = None  # record the last position
        self.vel = None

        # log name
        self.log_name = None
        self.log_file = None
        # rospy.Subscriber("/workpiece_pos", ImagePose, self.image_pose_callback)
        rospy.Subscriber("/twist", Twist, self.twist_callback)

        while not self.error_pos_status and not rospy.is_shutdown():
            # print(self.error_pos_status)
            continue

        rospy.loginfo("Detected, start to move.")

    def twist_callback(self, msg):
        self.twist[0] = msg.linear.x
        self.twist[1] = msg.linear.y
        self.twist[2] = msg.linear.z
        # self.image_pose(msg)

        if not (msg.angular.x == -1 and msg.angular.y == -1 and msg.angular.z == -1):
            self.error_pos_status = True

        status = None
        if not msg.angular.x == -1:
            pos_x = msg.linear.x
            pos_y = msg.linear.y
            pos_z = msg.linear.z
            self.last_pos = np.array([pos_x, pos_y, pos_z])
            status = True
        else:
            pos_x = self.last_pos[0]
            pos_y = self.last_pos[1]
            pos_z = self.last_pos[2]
            status = False

        if self.vel is None:
            self.kal_vel.set_initial_state(np.array([pos_x, pos_y, 0, 0]))
            self.vel = np.array([0.0, 0.0])
        else:
            self.vel = self.kal_vel.update(np.array([pos_x, pos_y]), status) * 5
            print(self.vel)

    def log_data(self):
        if self.error_pos_status:
            object_position = self.twist
            gripper_position, _, _ = self.get_leftgripper2base()
            timestamp = str(time.time())
            position = timestamp + ' ' + str(object_position[0]) + ' ' + str(object_position[1]) + ' ' \
                                 + str(object_position[2]) + ' ' + str(gripper_position[0]) + ' ' \
                                 + str(gripper_position[1]) + ' ' + str(gripper_position[2]) + '\n'

            log_name = self.get_log_name()
            if self.log_file is None:
                self.log_file = open(log_name, "w+")

            self.log_file.write(position)

    def get_log_name(self):
        if self.log_name is None:
            self.log_name = str(time.time())[0:-3] + '.txt'

        return self.log_name


    def image_pose(self, data):
        linear = data.linear
        position = numpy.array([linear.x, linear.y, 0.0])

        trans_camera, _, _ = self.get_leftcamera2base()
        depth = trans_camera[2] - pick_height - 0.04

        # x方向
        position[0] = -(position[0] - camera_cx_wrist) * depth / camera_fx_wrist
        # y方向
        position[1] = -(position[1] - camera_cy_wrist) * depth / camera_fy_wrist
        # 深度
        position[2] = depth

        self.workpiece_pos = numpy.array([position[1], position[0], position[2]])
        # print(self.workpiece_pos)
        position = numpy.append(position, 1)

        # 求出物件的空间坐标
        workpiece_marker = lookupTransform(self.tf_listener, '/left_hand_camera', '/base')
        trans_wp, rot, rot_euler = getTfFromMatrix(numpy.linalg.inv(workpiece_marker))
        translation_matrix = quaternion_to_trans_matrix(rot[0], rot[1], rot[2], rot[3],
                                                        trans_wp[0], trans_wp[1], trans_wp[2])
        base_tf = numpy.dot(translation_matrix, position)[0:3]
        self.workpiece_tf = base_tf

        # print(self.workpiece_tf)
        # 得到当前物件的角度
        self.theta = data.angular.x
        self.workpiece_tf_status = True

        if self.last_wp_tf is not None:
            current_time = 0.2
            self.ref_v[0] = (self.workpiece_tf[0] - self.last_wp_tf[0]) / current_time
            self.ref_v[1] = (self.workpiece_tf[1] - self.last_wp_tf[1]) / current_time
            self.rev_v_status = True
        self.last_wp_tf = copy.copy(self.workpiece_tf)
        self.last_time = time.clock()

    def set_neutral(self):
        """
        Sets both arms back into a neutral pose.
        """
        print("Moving to neutral pose...")
        self._left_arm.move_to_neutral()
        self._right_arm.move_to_neutral()

    def _reset_control_modes(self):
        rate = rospy.Rate(50)
        for _ in range(100):
            if rospy.is_shutdown():
                return False
            self._left_arm.exit_control_mode()
            self._right_arm.exit_control_mode()
            self._pub_rate.publish(100)  # 100Hz default joint state rate
            rate.sleep()
        return True

    def clean_shutdown(self):
        print("\nExiting example...")
        # return to normal
        self._reset_control_modes()
        self.log_file.close()
        # self.set_neutral()
        if not self._init_state:
            print("Disabling robot...")
            self._rs.disable()
        return True

    def get_joint_states(self):
        while True and not rospy.is_shutdown():
            data = rospy.wait_for_message('/robot/joint_states', JointState, timeout=4.0)
            if len(data.name) > 10:
                break
        name = data.name
        position = data.position

        joint_data = {}
        for i in range(len(name)):
            joint_data[name[i]] = position[i]

        q = [0] * 7
        for idx, name in enumerate(self._left_joint_names):
            q[idx] = joint_data[name]
        return q

    def get_gripper_position(self):
        return self.left.position()

    def get_leftgripper2base(self):
        self.tf_listener.waitForTransform('/left_gripper', '/base', rospy.Time(), rospy.Duration(4.0))
        gripper = lookupTransform(self.tf_listener, '/left_gripper', '/base')
        trans_gripper, rot_gripper, rot_euler_gripper = getTfFromMatrix(numpy.linalg.inv(gripper))
        return trans_gripper, rot_gripper, rot_euler_gripper

    def get_leftcamera2base(self):
        self.tf_listener.waitForTransform('/left_hand_camera', '/base', rospy.Time(), rospy.Duration(4.0))
        camera = lookupTransform(self.tf_listener, '/left_hand_camera', '/base')
        trans_camera, rot_camera, rot_euler_camera = getTfFromMatrix(numpy.linalg.inv(camera))
        return trans_camera, rot_camera, rot_euler_camera

    def get_error_pos(self, ref_down_vel):
        trans_gripper, _, _ = self.get_leftgripper2base()
        # print(trans_camera[0], trans_gripper[0], trans_camera[0] - trans_gripper[0], self.workpiece_tf)
        # print(trans_camera[1], trans_gripper[1], trans_camera[1] - trans_gripper[1], self.workpiece_tf)
        # error_pos = numpy.array([self.workpiece_pos[1] + 0.01,
        #                          self.workpiece_pos[0] - 0.03 * 1 / (
        #                                  1 + numpy.e ** ((dis_height - 0.01) * 200)),
        #                          ref_down_vel])

        diff_x = self.twist[0] - trans_gripper[0]
        diff_y = self.twist[1] - trans_gripper[1]
        error_pos = numpy.array([diff_x, diff_y, ref_down_vel])
        # self.workpiece_pos[0] = self.workpiece_pos[1] = 0.
        # theta = self.theta
        return error_pos

    def get_gmm_twist(self, error_pos, ratio=1.0):
        gmm_twist = numpy.asmatrix([0., 0., 0., 0., 0., 0.]).transpose()
        # 1. 使用控制率方法
        if self.vel is not None:
            u_x, u_y = control_rate(error_pos, numpy.array([self.vel[0], self.vel[1]]))
        else:
            u_x, u_y = control_rate(error_pos, numpy.array([0.0, 0.0]))
        # 2. 不使用控制率方法
        # u_x, u_y = error_pos[0], error_pos[1]
        gmm_twist[0] = u_y * ratio
        gmm_twist[1] = u_x * ratio
        gmm_twist[2] = error_pos[2]  # 控制夹爪下降的速度

        return gmm_twist

    @property
    def left_joint_names(self):
        return self._left_joint_names

    @property
    def left_arm(self):
        return self._left_arm


tracker = Tracker()


# define state Foo
class Follow(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=[])
        global tracker
        self.tracker = tracker
        self.ref_down_vel = 0.00

    def execute(self, userdata):
        rospy.loginfo('Executing state Follow')
        while not rospy.is_shutdown():
            trans_gripper, _, _ = self.tracker.get_leftgripper2base()
            dis_height = trans_gripper[2] - hang_height
            jadgger = get_jdagger_matrix(self.tracker.get_joint_states())

            # if 0.01 >= dis_height > 0.:
            #     return 'hangHeight'
            if dis_height <= 0:
                # 传入需要下降的速度
                error_pos = self.tracker.get_error_pos(self.ref_down_vel)
                rospy.loginfo("Follow error_pos: %f, %f, %f, pull up", error_pos[0], error_pos[1], error_pos[2])
                gmm_twist = self.tracker.get_gmm_twist(error_pos)
                # 向上提升
                # gmm_twist[2] = -0.03
            else:
                error_pos = self.tracker.get_error_pos(self.ref_down_vel)
                rospy.loginfo("Follow error_pos: %f, %f, %f", error_pos[0], error_pos[1], error_pos[2])
                gmm_twist = self.tracker.get_gmm_twist(error_pos)

            print("gmm_twist: ", gmm_twist[0:2].reshape(1, 2))
            qDot = jadgger * gmm_twist

            # 设定关节速度
            cmd = {}
            for idx, name in enumerate(self.tracker.left_joint_names):
                cmd[name] = qDot[idx]

            # 开启关闭用来记录示教和跟随数据
            self.tracker.left_arm.set_joint_velocities(cmd)
            self.tracker.log_data()
            self.tracker.rate.sleep()


# define state Bar
class Hangon(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['notHangHeight'])
        global tracker
        self.tracker = tracker
        self.ref_down_vel = 0.0  # 不要修改

    def execute(self, userdata):
        rospy.loginfo('Executing state BAR')
        while not rospy.is_shutdown():
            trans_gripper, _, _ = self.tracker.get_leftgripper2base()
            dis_height = trans_gripper[2] - hang_height
            jadgger = get_jdagger_matrix(self.tracker.get_joint_states())
            # print(dis_height)
            if dis_height < 0 or dis_height > 0.01:
                return 'notHangHeight'

            error_pos = self.tracker.get_error_pos(self.ref_down_vel)
            if abs(error_pos[0]) < converge and abs(error_pos[1]) < converge and abs(theta - 90.) <= 12:
                return 'achiveHeight'

            rospy.loginfo("Hangon error_pos: %f, %f; dis_height: %f.", error_pos[0], error_pos[1], dis_height)

            gmm_twist = self.tracker.get_gmm_twist(error_pos)
            print(gmm_twist[0], gmm_twist[1], gmm_twist[2])

            qDot = jadgger * gmm_twist
            # 设定关节速度
            cmd = {}
            for idx, name in enumerate(self.tracker.left_joint_names):
                cmd[name] = qDot[idx]

            self.tracker.left_arm.set_joint_velocities(cmd)
            self.tracker.rate.sleep()


class Grasp(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['success', 'failure'])
        global tracker
        self.tracker = tracker
        self.ref_grip_vel = 0.25

    def execute(self, userdata):
        while not rospy.is_shutdown():
            trans_gripper, _, _ = self.tracker.get_leftgripper2base()
            dis_height = trans_gripper[2] - pick_height
            jadgger = get_jdagger_matrix(self.tracker.get_joint_states())

            error_pos = self.tracker.get_error_pos(self.ref_grip_vel)
            rospy.loginfo("Grasp error_pos: %f, %f", error_pos[0], error_pos[1])
            gmm_twist = self.tracker.get_gmm_twist(error_pos)

            gmm_twist[1] = -0.05
            qDot = jadgger * gmm_twist
            # 设定关节速度
            cmd = {}
            for idx, name in enumerate(self.tracker.left_joint_names):
                cmd[name] = qDot[idx]

            self.tracker.left_arm.set_joint_velocities(cmd)
            rospy.sleep(0.3)

            self.tracker.left.command_position(0)
            rospy.sleep(1.0)
            gripper_position = self.tracker.get_gripper_position()
            # print(gripper_position)
            if gripper_position <= 5:
                self.tracker.left.command_position(100)
                rospy.sleep(1.0)
                return 'failure'
            else:
                return 'success'


class Pullup(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['end'])
        global tracker
        self.tracker = tracker
        self.ref_up_vel = -0.30

    def execute(self, userdata):
        while not rospy.is_shutdown():
            jadgger = get_jdagger_matrix(self.tracker.get_joint_states())
            gmm_twist = numpy.asmatrix([0., 0., 0., 0., 0., 0.]).transpose()
            gmm_twist[2] = self.ref_up_vel

            qDot = jadgger * gmm_twist
            # 设定关节速度
            cmd = {}
            for idx, name in enumerate(self.tracker.left_joint_names):
                cmd[name] = qDot[idx]

            self.tracker.left_arm.set_joint_velocities(cmd)
            rospy.sleep(0.3)

            return 'end'


# main
def main():
    # Create a SMACH state machine
    sm = smach.StateMachine(outcomes=['finish'])

    # Open the container
    with sm:
        # Add states to the container
        smach.StateMachine.add('Follow', Follow(), transitions={})
        smach.StateMachine.add('Hangon', Hangon(), transitions={'notHangHeight': 'Follow'})
        smach.StateMachine.add('Grasp', Grasp(), transitions={'success': 'Pullup', 'failure': 'Hangon'})
        smach.StateMachine.add('Pullup', Pullup(), transitions={'end': 'finish'})

    # Create and start the introspection server
    sis = smach_ros.IntrospectionServer('gmm_grasp', sm, '/SM_ROOT')
    sis.start()

    # Execute SMACH plan
    outcome = sm.execute()

    # Wait for ctrl-c to stop the application
    rospy.spin()
    sis.stop()


if __name__ == '__main__':
    main()
