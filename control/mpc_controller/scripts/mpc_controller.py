#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import tf
import numpy as np
import casadi as ca 

class MPCcontroller:
    def __init__(self):
        rospy.init_node("MPC_controller")
        rospy.loginfo("MPC Controller for Turtlebot3 is starting...")

        self.goal_x = rospy.get_param('~goal_x', 5.0)
        self.goal_y = rospy.get_param('~goal_y', 3.0)
        self.goal_theta = rospy.get_param('~goal_theta', 0.5)

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
        self.angle_tolerance = 0.1

        self.Q = np.diag([10.0, 12.0, 5.0])
        self.R = np.diag([10.0, 1.0])
        self.Qf = self.Q

        self.v_max = 0.6
        self.omega_max = np.pi / 4

        self.T = 30
        self.N_sim = 100

        rospy.loginfo(f"Q matrix: {self.Q.diagonal()}")
        rospy.loginfo(f"R matrix: {self.R.diagonal()}")
        rospy.loginfo(f"Control loop frequency: {self.rate_frequency} Hz (dt={self.dt:.3f}s)")

        self.solve_mpc()  # Only once

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, yaw) = tf.transformations.euler_from_quaternion(orientation_list)
        self.yaw = yaw

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def solve_mpc(self):
        x = ca.SX.sym("x")
        y = ca.SX.sym("y")
        theta = ca.SX.sym("theta")
        states = ca.vertcat(x, y, theta)
        self.n_states = states.size()[0]

        v = ca.SX.sym("v")
        omega = ca.SX.sym("omega")
        controls = ca.vertcat(v, omega)
        self.n_controls = controls.size()[0]

        rhs = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), omega)
        self.f = ca.Function("f", [states, controls], [rhs])

        U = ca.SX.sym("U", self.n_controls, self.T)
        X = ca.SX.sym("X", self.n_states, self.T + 1)
        self.P = ca.SX.sym("P", self.n_states * 2)  # initial + goal

        cost = 0
        g = [X[:, 0] - self.P[0:3]]

        for k in range(self.T):
            st = X[:, k]
            con = U[:, k]
            st_next = X[:, k + 1]
            f_value = self.f(st, con)
            st_next_euler = st + self.dt * f_value
            g.append(st_next - st_next_euler)

            state_error = st - self.P[3:6]
            cost += ca.mtimes([state_error.T, self.Q, state_error]) + ca.mtimes([con.T, self.R, con])

        final_error = X[:, self.T] - self.P[3:6]
        cost += ca.mtimes([final_error.T, self.Qf, final_error])

        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

        nlp_prob = {
            'f': cost,
            'x': opt_vars,
            'g': ca.vertcat(*g),
            'p': self.P
        }

        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 300}
        self.solver = ca.nlpsol("solver", "ipopt", nlp_prob, opts)

    def compute_action(self):
        state = np.array([self.x, self.y, self.yaw])
        goal = np.array([self.goal_x, self.goal_y, self.goal_theta])
        p = np.concatenate((state, goal))

        # Bounds
        lbx = []
        ubx = []

        for _ in range(self.T + 1):
            lbx.extend([-ca.inf, -ca.inf, -ca.inf])
            ubx.extend([ca.inf, ca.inf, ca.inf])

        for _ in range(self.T):
            lbx.extend([0.0, -self.omega_max])
            ubx.extend([self.v_max, self.omega_max])

        x0 = np.zeros((self.n_states * (self.T + 1) + self.n_controls * self.T, 1))

        sol = self.solver(x0=x0, lbx=lbx, ubx=ubx, lbg=0, ubg=0, p=p)
        sol_x = sol['x'].full().flatten()

        u = sol_x[self.n_states * (self.T + 1): self.n_states * (self.T + 1) + self.n_controls]
        v, omega = u[0], u[1]

        # Goal check
        distance = np.linalg.norm([self.x - self.goal_x, self.y - self.goal_y])
        angle_error = self.normalize_angle(self.goal_theta - self.yaw)

        cmd = Twist()
        if distance < self.distance_tolerance and abs(angle_error) < self.angle_tolerance:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            rospy.loginfo("🎯 Goal Reached!")
        else:
            cmd.linear.x = v
            cmd.angular.z = omega

        rospy.loginfo(f"Current Pos: ({self.x:.2f}, {self.y:.2f}), Yaw: {self.yaw:.2f}")
        rospy.loginfo(f"Distance to goal: {distance:.3f}, Angle error: {angle_error:.3f}")
        return cmd

    def run(self):
        while not rospy.is_shutdown():
            cmd = self.compute_action()
            self.cmd_vel_pub.publish(cmd)
            self.rate.sleep()


if __name__ == '__main__':
    try:
        controller = MPCcontroller()
        controller.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("MPC Controller node shut down.")
