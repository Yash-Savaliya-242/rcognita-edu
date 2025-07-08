#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import tf
import numpy as np
from scipy.linalg import solve_discrete_are

class LQRcontroller:
    def __init__(self):
        rospy.init_node("LQR_controller")
        rospy.loginfo("LQR Controller for Turtlebot3 is starting...")

        self.goal_x = rospy.get_param('~goal_x', 2.0)
        self.goal_y = rospy.get_param('~goal_y', 3.0)
        self.goal_theta = rospy.get_param('~goal_theta', 0.5)  # NEW: desired final orientation
        rospy.loginfo(f"Goal set to: ({self.goal_x}, {self.goal_y}, θ={self.goal_theta} rad)")

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)

        self.rate_frequency = 10  # Hz
        self.rate = rospy.Rate(self.rate_frequency)
        self.dt = 1.0 / self.rate_frequency

        self.distance_tolerance = 0.1
        self.angle_tolerance = 0.1  # NEW: tolerance for reaching final yaw

        self.Q = np.diag([10.0, 12.0, 5.0])
        self.R = np.diag([10.0, 1.0])
        rospy.loginfo(f"LQR Q matrix: {self.Q.diagonal()}")
        rospy.loginfo(f"LQR R matrix: {self.R.diagonal()}")
        rospy.loginfo(f"Control loop frequency: {self.rate_frequency} Hz (dt={self.dt:.3f}s)")

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, yaw) = tf.transformations.euler_from_quaternion(orientation_list)
        self.yaw = yaw

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def compute_action(self):
        dt = self.dt

        dx_global = self.x - self.goal_x
        dy_global = self.y - self.goal_y
        e_theta = self.normalize_angle(self.goal_theta - self.yaw)  # NEW: orientation error

        X_error_lqr = np.array([dx_global, dy_global, e_theta])
        v0 = 0.001  # small linearization velocity

        A_lqr = np.array([
            [1, 0, -v0 * math.sin(self.yaw) * dt],
            [0, 1, v0 * math.cos(self.yaw) * dt],
            [0, 0, 1]
        ])

        B_lqr = np.array([
            [math.cos(self.yaw) * dt, 0],
            [math.sin(self.yaw) * dt, 0],
            [0, dt]
        ])

        try:
            P = solve_discrete_are(A_lqr, B_lqr, self.Q, self.R)
            K = np.linalg.inv(self.R + B_lqr.T @ P @ B_lqr) @ (B_lqr.T @ P @ A_lqr)
        except np.linalg.LinAlgError as e:
            rospy.logwarn(f"LQR solve_discrete_are failed: {e}. Returning zero controls.")
            return Twist()

        u = -K @ X_error_lqr
        cmd_linear_x = np.clip(u[0], -2.0, 2.0)
        cmd_angular_z = np.clip(u[1], -3.14, 3.14)

        distance = np.linalg.norm([dx_global, dy_global])
        angle_error = e_theta

        rospy.loginfo(f"Current Pos: ({self.x:.2f}, {self.y:.2f}), Yaw: {self.yaw:.2f}")
        rospy.loginfo(f"Goal Errors: dx={dx_global:.2f}, dy={dy_global:.2f}, d_yaw={e_theta:.2f}")
        rospy.loginfo(f"Distance to goal: {distance:.3f}, Angle error: {angle_error:.3f}")

        cmd = Twist()
        if distance > self.distance_tolerance:
            # if abs(angle_error) > self.angle_tolerance:
            # # Phase 1: Rotate toward the goal
            #     cmd.linear.x = 0.0
            #     cmd.angular.z =cmd_angular_z
            #     #  np.clip(1.5 * angle_error, -1.0, 1.0)  # Proportional turn-in-place
            # else:
            cmd.linear.x = cmd_linear_x
            cmd.angular.z = cmd_angular_z            
        else:
        # Final orientation correction after reaching goal
            if abs(e_theta) > self.angle_tolerance:
                cmd.linear.x = 0.0
                cmd.angular.z = np.clip(1.5 * e_theta, -1.0, 1.0)
            else:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                rospy.loginfo("🎯 Goal Reached with Final Orientation!")

        return cmd

    def run(self):
        while not rospy.is_shutdown():
            cmd = self.compute_action()
            self.cmd_vel_pub.publish(cmd)
            self.rate.sleep()

if __name__ == '__main__':
    try:
        controller = LQRcontroller()
        controller.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("LQR Controller node shut down.")






# new code





