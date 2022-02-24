#!/usr/bin/env python
# coding=utf-8

import numpy as np
import rospy
from geometry_msgs.msg import Twist


class KalmanFilterVel:
    def __init__(self):
        dt = 0.06
        self.A = np.array([[1,  0,  dt, 0],
                           [0,  1,  0,  dt],
                           [0,  0,  1,  0],
                           [0,  0,  0,  1]])

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


class VelCallback:
    def __init__(self):
        rospy.Subscriber("/twist", Twist, self.twist_callback)
        self.kal_vel = KalmanFilterVel()
        self.last_pos = None
        self.kalman_status = False

    def twist_callback(self, msg):
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

        if self.kalman_status is False:
            self.kal_vel.set_initial_state(np.array([pos_x, pos_y, 0, 0]))
            self.kalman_status = True
        else:
            vel = self.kal_vel.update(np.array([pos_x, pos_y]), status)
            print(vel)


def main():
    rospy.init_node('vel_test', anonymous=True)
    velcb = VelCallback()
    rospy.spin()


if __name__ == '__main__':
    main()
