#!/usr/bin/env python



import argparse
import math
import random

import numpy
import baxter_left_kinematics as blk

import rospy

from std_msgs.msg import (
    UInt16,
)
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist

import baxter_interface

from baxter_interface import CHECK_VERSION


class Tracker(object):



    def __init__(self):
        """
        Tracks a user's fist using the wrist camera.
        """
        self._pub_rate = rospy.Publisher('robot/joint_state_publish_rate',
                                         UInt16, queue_size=10)
        self._left_arm = baxter_interface.limb.Limb("left")
        self._right_arm = baxter_interface.limb.Limb("right")
        self._left_joint_names = self._left_arm.joint_names()
        self._right_joint_names = self._right_arm.joint_names()

        self.joint_states = {}
        self.twist = numpy.asmatrix([0.,0.,0.,0.,0.,0.]).transpose()

        rospy.Subscriber("/robot/joint_states", JointState, self.joint_state_callback)

        rospy.Subscriber("/twist", Twist, self.twist_callback)

        # control parameters
        self._rate = 100.0  # Hz

        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()

        # set joint state publishing to 50Hz
        self._pub_rate.publish(self._rate)

    def twist_callback(self, msg):
        
        self.twist[3] = msg.angular.x
        self.twist[4] = msg.angular.y
        self.twist[5] = msg.angular.z
        rospy.loginfo("twist: %s", str(self.twist))

    def joint_state_callback(self, msg):
        #self.lock.acquire()
        self.joint_states = dict(zip(msg.name, msg.position))
        #rospy.loginfo("Joint states: %s", str(self.joint_states))
        #self.lock.release()

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
        #return to normal
        self._reset_control_modes()
        self.set_neutral()
        if not self._init_state:
            print("Disabling robot...")
            self._rs.disable()
        return True

    def track(self):
        """

        """
        self.set_neutral()
        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            #obtain the joint states as an array
            rospy.loginfo("Joint states: %s", str(self.joint_states))
            #rospy.loginfo("Joint names: %s", str(self._left_joint_names))
            q = [0]*7
            for idx, name in enumerate(self._left_joint_names):
                #rospy.loginfo("name: %s", name)
                name
                q[idx] = self.joint_states[name]
            rospy.loginfo("Joint angles: %s", str(q))
            #obtain the current Jacobian (geometric will do fine)
            J = numpy.asmatrix(blk.J[6](q))
            #transform the Jacobian to the end-effector frame
            T = numpy.asmatrix(blk.FK[6](q))
            R_n_0 = T[numpy.ix_([0,1,2],[0,1,2])].transpose()
            R_c_n = numpy.matrix([[ 0, 1, 0],
                                  [-1, 0, 0],
                                  [ 0, 0, 1]])
            R_c_0 = R_c_n*R_n_0
            Z = numpy.asmatrix(numpy.zeros(9).reshape(3,3))
            X1 = numpy.concatenate((R_c_0, Z), axis = 1)
            X2 = numpy.concatenate((Z, R_c_0), axis = 1)
            X_c_0 = numpy.concatenate((X1, X2), axis = 0)
            J = X_c_0*J
            Jt = J.transpose()
            # damping factor for pseudo inverse
            delta = 0.001
            I = numpy.eye(6)
            #rospy.loginfo("Jacobian: %s", str(J))
            #command a desired twist and calculate needed joint angles
            #twist = numpy.asmatrix([0,0,0,0.1,-0.1,0]).transpose()
            Jdagger = Jt*numpy.linalg.pinv(J*Jt + delta*I)
            #rospy.loginfo("Jdagger: %s", str(Jdagger))
            if self.twist[3] != 0.0:
		self.twist[2] = 0.1
	    else:
		self.twist[2] = 0.0
            qDot = Jdagger*self.twist
            #qDot = [0,0,0,0,0,0.1,0]
            rospy.loginfo("commanded velocity: %s", str(qDot))
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
