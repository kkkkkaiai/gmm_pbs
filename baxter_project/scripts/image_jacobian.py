#!/usr/bin/env python
import rospy
import numpy as np
import math as m
from tld_msgs.msg import BoundingBox
from geometry_msgs.msg import Twist

half_width = 320
half_height = 200
u = 0   # target location in pixel
w = 0
confidence = 0

f = 1000   # focal_length in pixel
alpha = 1   # tuning parameter

def callback(data):
    global u
    global w
    global confidence
    u = data.x-half_width
    w = data.y-half_height
    confidence = data.confidence

def calculate_command():
    global u
    global w
    global alpha
    denominator = 1.0/(m.pow(f,2)+m.pow(u,2)+m.pow(w,2))
    if confidence==0:
        angular_command = np.matrix([[0],[0],[0]])
    else:
        angular_command = denominator*np.matrix([[-alpha*w*f],[alpha*u*f],[2*alpha*u*w]])
    return angular_command

def listener():
    rospy.init_node('image_jacobian', anonymous=True)
    rospy.Subscriber("/tld_tracked_object", BoundingBox, callback)
    pub = rospy.Publisher("/twist", Twist, queue_size=1)

    rate = rospy.Rate(30)   # 10hz
    while not rospy.is_shutdown():
        angular_command = calculate_command()
        
        #print angular_command
        msg = Twist()
        msg.angular.x = -angular_command.item(0)
        msg.angular.y = -angular_command.item(1)
        msg.angular.z = -angular_command.item(2)
        pub.publish(msg)
        rate.sleep()

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()

