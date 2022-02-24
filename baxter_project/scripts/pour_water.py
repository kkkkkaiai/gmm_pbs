#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import rospy
import copy
from struct import *

import baxter_interface
import baxter_external_devices

from baxter_interface import CHECK_VERSION
import baxter_left_kinematics as blk

from sensor_msgs.msg import JointState

import socket
import time

MaxBytes = 11


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server.settimeout(600)
host = '172.20.172.204'
host = '47.99.152.116'

port = 8800
port = 10067
server.bind((host, port))
server.listen(1)


class BaxterState:
    def __init__(self):
        self.left = baxter_interface.Limb('left')
        self.right = baxter_interface.Limb('right')
        self.grip_left = baxter_interface.Gripper('left', CHECK_VERSION)
        self.grip_right = baxter_interface.Gripper('right', CHECK_VERSION)
        self.lj = self.left.joint_names()
        self.rj = self.right.joint_names()
        self.joint_state = {}
        rospy.Subscriber("/robot/joint_states", JointState, self.joint_state_callback)

    def joint_state_callback(self, msg):
        # self.lock.acquire()
        self.joint_states = dict(zip(msg.name, msg.position))

    def get_joint_state(self):
        return self.joint_state


def pour_water(flag, baxter_state):
    joint_position_1 = {'left_e0': 0.11926700625809092, 'left_e1': 1.7598594589015408, 'left_s0': -0.7750437930791053,
                        'left_s1': -0.3846456825622675, 'left_w0': 0.06711165946998685, 'left_w1': -1.3203739631723699,
                        'left_w2': -0.1902136176977913}


    joint_position_2 = {'left_e0': 0.4088058799714628, 'left_e1': 1.9017526817809416, 'left_s0': -1.196888509747594,
                        'left_s1': -0.7236554366849439, 'left_w0': -0.27765052260725986, 'left_w1': -0.9391797373828445,
                        'left_w2': -0.21360682471304387}

    # right
    joint_position_3 = {'left_e0': 0.023776702212223912, 'left_e1': 1.679708962734528, 'left_s0': -1.284708909854034,
                        'left_s1': -0.681470965018095, 'left_w0': -0.6929758209272356, 'left_w1': -1.0618982004136777,
                        'left_w2': 0.17180584824316633}

    joint_position_4 = {'left_e0': 0.4084223847744914, 'left_e1': 1.5738642883704346, 'left_s0': -1.3437671701876224,
                        'left_s1': -0.184461189743221, 'left_w0': -0.5388107517447516, 'left_w1': -1.1332283070503495,
                        'left_w2': -0.3604854851530722}

    joint_position_5 = {'left_e0': 0.1836941993492783, 'left_e1': 1.267451625990323, 'left_s0': -0.872451573109829,
                        'left_s1': 0.08321845774278369, 'left_w0': -0.11159710231866385, 'left_w1': -1.2755050251267215,
                        'left_w2': -0.24121847889498133}

    if flag == 2:
        print('initial pose', flag)
        baxter_state.left.move_to_joint_positions(joint_position_4)
    elif flag == 6 or flag == 7:
        print('move up,', flag)
        baxter_state.left.move_to_joint_positions(joint_position_2)
    elif flag == 5:
        print('move right,', flag)
        baxter_state.left.move_to_joint_positions(joint_position_3)
    elif flag == 8:
        print('move to prepare position,', flag)
        baxter_state.left.move_to_joint_positions(joint_position_4)
    elif flag == 1:
        joint = 'left' + "_w2"
        joint_position_4[joint] = 1.593806038612945
        baxter_state.left.move_to_joint_positions(joint_position_4)
    elif flag == 3:
        joint = 'left' + "_w2"
        joint_position_4[joint] = 0.07056311624272903
        baxter_state.left.move_to_joint_positions(joint_position_4)
    # elif flag == 7:
    #     joint = 'left' + "_w2"
    #     joint_position_4[joint] = -0.5384272565477802
    #     baxter_state.left.move_to_joint_positions(joint_position_4)
    # elif flag == 8:
        baxter_state.left.move_to_joint_positions(joint_position_5)
        print('gripper open')
        baxter_state.grip_left.open()
        time.sleep(10)
    # elif flag == 9:
        



def receive_data(baxter_state):
    st = Struct('BBBBBBBBBBc')
    pre_st = Struct('BBBBBBBB')
    try:
        client, addr = server.accept()
        print(addr, " connected")
        data = client.recv(MaxBytes)
        data = pre_st.unpack(data)
        print(data)
        data = list(data)
        data[5] += 1
        data = pre_st.pack(*data)
        client.send(data)

        while True:
            print('==================================')
            print('wait for client')
            # client, addr = server.accept()
            # print(addr, " connected")
            data = client.recv(MaxBytes)
            if not data:
                print('empty data, sleep 3 s')
                time.sleep(5)
                continue
            # print(data)
            data = st.unpack(data)

            flag = data[8]
            print('action: ', flag)
            pour_water(flag, baxter_state)

            localTime = time.asctime(time.localtime(time.time()))
            # print(localTime, ' receive the length of data:', len(data))
            send_data = list(copy.copy(data))
            send_data[8] += 1
            result = st.pack(*send_data)
            client.send(result)
            time.sleep(1)
            print('finish action: ', flag)
    except BaseException as e:
        print("Exception：")
        print(repr(e))
    finally:
        server.close()
        print("closed")


def main():
    print("Initializing node... ")
    rospy.init_node("rsdk_ee_position")
    print("Getting robot state... ")
    rs = baxter_interface.RobotEnable(CHECK_VERSION)
    init_state = rs.state().enabled

    def clean_shutdown():
        print("\nExiting example...")
        if not init_state:
            print("Disabling robot...")
            rs.disable()

    rospy.on_shutdown(clean_shutdown)

    print("Enabling robot... ")
    rs.enable()

    _baxter = BaxterState()
    receive_data(_baxter)

    # pour_water(1, _baxter)
    # pour_water(2, _baxter)
    # pour_water(3, _baxter)
    # pour_water(4, _baxter)
    # time.sleep(1)
    # pour_water(5, _baxter)
    # pour_water(6, _baxter)
    # pour_water(7, _baxter)
    # pour_water(8, _baxter)
    # pour_water(9, _baxter)


if __name__ == '__main__':
    main()
