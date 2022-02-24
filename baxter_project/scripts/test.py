#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import rospy
import copy
from struct import *

from sensor_msgs.msg import JointState
import baxter_interface
import baxter_external_devices

from baxter_interface import CHECK_VERSION

joint_position_1 = {'left_e0': 0.11926700625809092, 'left_e1': 1.7598594589015408, 'left_s0': -0.7750437930791053,
                        'left_s1': -0.3846456825622675, 'left_w0': 0.06711165946998685, 'left_w1': -1.3203739631723699,
                        'left_w2': -0.1902136176977913}

joint_position_2 = {'left_e0': 0.18983012250081996, 'left_e1': 1.8538157821595225, 'left_s0': -0.9269078910797612,
                    'left_s1': -0.9790632378678653, 'left_w0': 0.08551942892461181, 'left_w1': -0.884339924215941,
                    'left_w2': -0.2044029399857314}

# right
joint_position_3 = {'left_e0': 0.18791264651596318, 'left_e1': 1.3230584295511694, 'left_s0': -1.2406069622023284,
                    'left_s1': -0.7263399030637434, 'left_w0': -0.6668981475331837, 'left_w1': -0.5426457037144651,
                    'left_w2': 0.3029612056073692}

joint_position_4 = {'left_e0': 0.4291311254109445, 'left_e1': 1.1455001533534328, 'left_s0': -1.565810889234036,
                    'left_s1': -0.07133010663667173, 'left_w0': -0.9380292517919305, 'left_w1': -0.8287331206550947,
                    'left_w2': 0.07056311624272903}

joint_position_5 = {'left_e0': 0.1836941993492783, 'left_e1': 1.267451625990323, 'left_s0': -0.872451573109829,
                    'left_s1': 0.08321845774278369, 'left_w0': -0.11159710231866385, 'left_w1': -1.2755050251267215,
                    'left_w2': -0.24121847889498133}


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
    _baxter.left.move_to_joint_positions(joint_position_1)


if __name__ == '__main__':
    main()