#!/usr/bin/env python

from __future__ import print_function

import threading
import numpy as np

import roslib; roslib.load_manifest('tele_operate')
import rospy

from geometry_msgs.msg import Twist

import sys, select, termios, tty

msg = """
Reading from the keyboard  and Publishing to Twist!
---------------------------
Moving around:

        w     
    
   a    s    d

anything else : stop

u/i : increase/decrease only linear speed by 10%
o/p : increase/decrease only angular speed by 10%

CTRL-C to quit
"""

moveBindings = {
        'w':(3,0,0,0),
        's':(-1,0,0,0),
        'q':(1,0,0,1),
        'e':(1,0,0,-1),
        'a':(-1,0,0,1),
        'd':(-1,0,0,-1),
        'z':(0,0,0,1),
        'c':(0,0,0,-1),
    }

speedBindings={
        'u':(1.1,1),
        'i':(.9,1),
        'o':(1,1.1),
        'p':(1,.9),
    }

class PublishThread(threading.Thread):
    def __init__(self, rate):
        super(PublishThread, self).__init__()
        self.publisher = rospy.Publisher('cmd_vel', Twist, queue_size = 1)
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.th = 0.0
        self.speed = 0.0
        self.turn = 0.0
        self.forward = 0.0
        self.steering = 0.0
        self.forward_pre = 0.0
        self.steering_pre = 0.0
        self.forward_LPF = 0.0
        self.steering_LPF = 0.0
        self.condition = threading.Condition()
        self.done = False

        # Set timeout to None if rate is 0 (causes new_message to wait forever
        # for new data to publish)
        if rate != 0.0:
            self.timeout = 1.0 / rate
        else:
            self.timeout = None

        self.start()

    def wait_for_subscribers(self):
        i = 0
        while not rospy.is_shutdown() and self.publisher.get_num_connections() == 0:
            if i == 4:
                print("Waiting for subscriber to connect to {}".format(self.publisher.name))
            rospy.sleep(0.5)
            i += 1
            i = i % 5
        if rospy.is_shutdown():
            raise Exception("Got shutdown request before subscribers connected")

    def LPF(self, data, data_pre, freq):
        delT = 0.05
        K_LPF = freq*2.0*3.141592*delT/(1.0 + freq*2.0*3.141592*delT)
        data_LPF = (data - data_pre) * K_LPF + data_pre
        return data_LPF

    def update(self, x, y, z, th, speed, turn):
        self.condition.acquire()
        self.x = x
        self.y = y
        self.z = z
        self.th = th
        self.speed = speed
        self.turn = turn

        # Notify publish thread that we have a new message.
        self.condition.notify()
        self.condition.release()

    def stop(self):
        self.done = True
        self.update(0, 0, 0, 0, 0, 0)
        self.join()

    def run(self):
        twist = Twist()
        while not self.done:
            self.condition.acquire()
            # Wait for a new message or timeout.
            self.condition.wait(self.timeout)

            # Copy state into twist message.
            self.forward = self.x * self.speed
            self.forward_LPF = self.LPF(self.forward, self.forward_pre, 0.3)
            self.forward_pre = self.forward_LPF

            self.steering = self.th * self.turn
            self.steering_LPF = self.LPF(self.steering, self.steering_pre, 1.0)
            self.steering_pre = self.steering_LPF

            if np.fabs(self.forward_LPF) < 0.01:
                self.forward_LPF = 0.0
            if np.fabs(self.steering_LPF) < 0.001:
                self.steering_LPF = 0.0

            twist.linear.x = self.forward_LPF
            twist.linear.y = 0
            twist.linear.z = 0
            twist.angular.x = 0
            twist.angular.y = 0
            twist.angular.z = self.steering_LPF

            self.condition.release()

            # Publish.
            self.publisher.publish(twist)

        # Publish stop message when thread exits.
        twist.linear.x = 0
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = 0
        self.publisher.publish(twist)


def getKey(key_timeout):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], key_timeout)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def vels(speed, turn):
    return "currently:\tspeed %s\tturn %s " % (speed,turn)

if __name__=="__main__":
    settings = termios.tcgetattr(sys.stdin)

    rospy.init_node('tele_operate')

    speed = rospy.get_param("~speed", 3.0)
    turn = rospy.get_param("~turn", 0.4)
    repeat = rospy.get_param("~repeat_rate", 20.0)
    key_timeout = rospy.get_param("~key_timeout", 0.6)
    if key_timeout == 0.0:
        key_timeout = None

    pub_thread = PublishThread(repeat)

    x = 0
    y = 0
    z = 0
    th = 0
    status = 0

    try:
        pub_thread.wait_for_subscribers()
        pub_thread.update(x, y, z, th, speed, turn)

        print(msg)
        print(vels(speed,turn))
        while(1):
            key = getKey(key_timeout)
            if key in moveBindings.keys():
                x = moveBindings[key][0]
                y = 0.0
                z = 0.0
                th = moveBindings[key][3]
            elif key in speedBindings.keys():
                speed = speed * speedBindings[key][0]
                turn = turn * speedBindings[key][1]

                print(vels(speed,turn))
                if (status == 14):
                    print(msg)
                status = (status + 1) % 15
            else:
                # Skip updating cmd_vel if key timeout and robot already
                # stopped.
                if key == '' and x == 0 and y == 0 and z == 0 and th == 0:
                    continue
                x = 0
                y = 0
                z = 0
                th = 0
                if (key == '\x03'):
                    break
 
            pub_thread.update(x, y, z, th, speed, turn)

    except Exception as e:
        print(e)

    finally:
        pub_thread.stop()

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
