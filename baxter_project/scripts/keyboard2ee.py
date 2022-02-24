#!/usr/bin/env python

import argparse
import numpy as np
import rospy

import baxter_interface
import baxter_external_devices

from baxter_interface import CHECK_VERSION
import baxter_left_kinematics as blk

from sensor_msgs.msg import JointState

class baxter_state():
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
        #self.lock.acquire()
        self.joint_states = dict(zip(msg.name, msg.position))
    
    def get_joint_state(self):
        return self.joint_state

def map_keyboard(robot):
    right = baxter_interface.Limb('right')
    grip_right = baxter_interface.Gripper('right', CHECK_VERSION)
    rj = right.joint_names()
    left = baxter_state().left
    grip_left = baxter_state().grip_left
    lj = left.joint_names()

    def set_j(limb, joint_name, delta):
        current_position = limb.joint_angle(joint_name)
        joint_command = {joint_name: current_position + delta}
        limb.set_joint_positions(joint_command)

    def calc(twist, arm):
        joint_states = left.joint_angles()
        # print(joint_states)
        _left_joint_names = arm.joint_names()
        q = [0]*7
        for idx, name in enumerate(_left_joint_names):
            #rospy.loginfo("name: %s", name)
            q[idx] = joint_states[name]
        # rospy.loginfo("Joint angles: %s", str(q))
        #obtain the current Jacobian (geometric will do fine)
        J = np.asmatrix(blk.J[6](q))
        #transform the Jacobian to the end-effector frame
        T = np.asmatrix(blk.FK[6](q))
        R_n_0 = T[np.ix_([0,1,2],[0,1,2])].transpose()
        R_c_n = np.matrix([[ 0, 1, 0],
                                [-1, 0, 0],
                                [ 0, 0, 1]])
        R_c_0 = R_c_n*R_n_0
        Z = np.asmatrix(np.zeros(9).reshape(3,3))
        X1 = np.concatenate((R_c_0, Z), axis = 1)
        X2 = np.concatenate((Z, R_c_0), axis = 1)
        X_c_0 = np.concatenate((X1, X2), axis = 0)
        J = X_c_0*J
        Jt = J.transpose()
        # damping factor for pseudo inverse
        delta = 0.001
        I = np.eye(6)
        #rospy.loginfo("Jacobian: %s", str(J))
        #command a desired twist and calculate needed joint angles
        #twist = numpy.asmatrix([0,0,0,0.1,-0.1,0]).transpose()
        Jdagger = Jt*np.linalg.pinv(J*Jt + delta*I)
        #rospy.loginfo("Jdagger: %s", str(Jdagger))
        # if twist[3] != 0.0:
        #     twist[2] = 0.1
        # else:
        #     twist[2] = 0.0

        qDot = Jdagger*twist
        # rospy.loginfo("commanded velocity: %s", str(qDot))
        cmd = {}
        for idx, name in enumerate(_left_joint_names):
            cmd[name] = qDot[idx]
        # print(cmd)
        return cmd
        #arm.set_joint_velocities(cmd)

    def set_le(limb, ):
        pass
    
    ee_ratio = 0.5
    ratio = 0.1
    # twist = [y, x, z, roll, pitch, yaw]
    bindings = {
    #   key: (function, args, description)
        'g': (grip_left.close, [], "left: gripper close"),
        't': (grip_left.open, [], "left: gripper open"),
        '/': (grip_left.calibrate, [], "left: gripper calibrate"),

        
        # 'r': (set_j, [right, rj[2], 0.1], "right_e0 increase"),
        # 'q': (set_j, [right, rj[2], -0.1], "right_e0 decrease"),
        # 'e': (set_j, [right, rj[3], 0.1], "right_e1 increase"),
        'w': (calc, [np.asmatrix([0.,1.*ratio,0.,0.,0.,0.]).transpose(), left], "ee_x front"),
        'a': (calc, [np.asmatrix([1.*ratio,0.,0.,0.,0.,0.]).transpose(), left], "ee_x back"),
        's': (calc, [np.asmatrix([0.,-1.*ratio,0.,0.,0.,0.]).transpose(), left], "ee_y left"),
        'd': (calc, [np.asmatrix([-1.*ratio,0.,0.,0.,0.,0.]).transpose(), left], "ee_y right"),
        'z': (calc, [np.asmatrix([0.,0.,1.*ratio,0.,0.,0.]).transpose(), left], "ee_z down"),
        'x': (calc, [np.asmatrix([0.,0.,-1.*ratio,0.,0.,0.]).transpose(), left], "ee_z up"),
        '1': (calc, [np.asmatrix([0.,0.,0.,1.*ee_ratio,0.,0.]).transpose(), left], "ee pitch"),
        '3': (calc, [np.asmatrix([0.,0.,0.,-1.*ee_ratio,0.,0.]).transpose(), left], "ee pitch"),
        'f': (calc, [np.asmatrix([0.,0.,0.,0.,1.*ee_ratio,0.]).transpose(), left], "ee_yaw"),
        'r': (calc, [np.asmatrix([0.,0.,0.,0.,-1.*ee_ratio,0.]).transpose(), left], "ee_yaw"),
        'q': (calc, [np.asmatrix([0.,0.,0.,0.,0.,1.*ee_ratio]).transpose(), left], "ee_roll"),
        'e': (calc, [np.asmatrix([0.,0.,0.,0.,0.,-1.*ee_ratio]).transpose(), left], "ee_roll"),

        # 'v': (set_j, [right, rj[6], 0.1], "right_w2 increase"),
        # 'z': (set_j, [right, rj[6], -0.1], "right_w2 decrease"),
        # 'c': (grip_right.close, [], "right: gripper close"),
        # 'x': (grip_right.open, [], "right: gripper open"),
        # 'b': (grip_right.calibrate, [], "right: gripper calibrate"),
     }
    done = False
    print("Controlling joints. Press ? for help, Esc to quit.")
    while not done and not rospy.is_shutdown():
        c = baxter_external_devices.getch()
        if c:
            #catch Esc or ctrl-c
            if c in ['\x1b', '\x03']:
                done = True
                rospy.signal_shutdown("Example finished.")
            elif c in bindings:
                cmd = bindings[c]
                result = cmd[0](*cmd[1])
                #expand binding to something like "set_j(right, 's0', 0.1)"
                if cmd[0] == calc:
                    left.set_joint_velocities(result)
                print("command: %s" % (cmd[2],))
            else:
                print("key bindings: ")
                print("  Esc: Quit")
                print("  ?: Help")
                for key, val in sorted(bindings.items(),
                                       key=lambda x: x[1][2]):
                    print("  %s: %s" % (key, val[2]))
        # cmd = {}
        # zero_twist = [0,0,0,0,0,0,0]
        # _left_joint_names = left.joint_names()
        # for idx, name in enumerate(_left_joint_names):
        #     cmd[name] = zero_twist[idx]
        # left.set_joint_velocities(cmd)


def main():
    """RSDK Joint Position Example: Keyboard Control

    Use your dev machine's keyboard to control joint positions.

    Each key corresponds to increasing or decreasing the angle
    of a joint on one of Baxter's arms. Each arm is represented
    by one side of the keyboard and inner/outer key pairings
    on each row for each joint.
    """
    epilog = """
See help inside the example with the '?' key for key bindings.
    """
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__,
                                     epilog=epilog)
    parser.parse_args(rospy.myargv()[1:])

    print("Initializing node... ")
    rospy.init_node("rsdk_joint_position_keyboard")
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

    _baxter = baxter_state()
    map_keyboard(_baxter)
    print("Done.")


if __name__ == '__main__':
    main()
