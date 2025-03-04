#!/usr/bin/python3

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from scipy.optimize import minimize

# Constants
m = 2.35  # Mass of the vehicle (kg)
Iz = 0.045  # Moment of inertia around z-axis (kg*m^2)
Lf = 0.11372  # Distance from the center of gravity to the front axle (m)
Lr = 0.14328  # Distance from the center of gravity to the rear axle (m)
Bf, Cf, Df = 2.0, 1.2, -1.0
Br, Cr, Dr = 2.0, 1.2, -1.0

# Blending speed thresholds
v_blend_min = 0.1
v_blend_max = 2.5

# Define parameters
dt = 0.1  # time step

circle_radius = 10.0  # Radius of the circle
circle_center = np.array([0, 0])  # Center of the circle

class DriftController(Node):
    def __init__(self):
        super().__init__('drift_controller')
        
        # Subscribe to vehicle pose
        self.pose_subscription = self.create_subscription(
            Odometry,
            '/vehicle/pose',  # Update with the correct topic name
            self.pose_callback,
            10)
        
        # Publisher for control commands
        self.control_publisher = self.create_publisher(
            Twist,
            '/vehicle/cmd_vel',  # Update with the correct topic name
            10)
        
        # Initial state
        self.state = np.array([10.0, 0, np.pi/2, 0, 0, 0, 0])
        self.controls = []
        self.time = [0]
        self.targets = []

    def pose_callback(self, msg):
        # Extract pose information from the message
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        r = msg.twist.twist.angular.z
        
        # Update state
        self.state = np.array([x, y, yaw, vx, vy, r, 0])
        
        # Run MPC
        self.run_mpc()

    def quaternion_to_yaw(self, orientation):
        # Convert quaternion to yaw angle
        x = orientation.x
        y = orientation.y
        z = orientation.z
        w = orientation.w
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
        return yaw

    def run_mpc(self):
        N = 10  # Prediction horizon
        U0 = [0.0, 0.0] * N  # Initial guess for controls
        bounds = [(-5.0, 5.0), (-0.698, 0.698)] * N  # Constraints for controls

        # Get current target on the circle
        t = len(self.time)
        target = self.circle_target(t * dt, circle_radius, circle_center)
        self.targets.append(target)

        result = minimize(
            self.mpc_cost, U0, args=(N, self.state, target),
            bounds=bounds, method='SLSQP'
        )
        if not result.success:
            self.get_logger().error("Optimization failed!")
            return

        # Apply the first control input
        U_opt = result.x
        control = U_opt[:2]
        self.state = self.drift_model(self.state, control, dt)
        self.controls.append(control)
        self.time.append((t + 1) * dt)

        # Publish control command
        self.publish_control(control)

    def publish_control(self, control):
        # Create Twist message
        twist_msg = Twist()
        twist_msg.linear.x = control[0]
        twist_msg.angular.z = control[1]
        
        # Publish the message
        self.control_publisher.publish(twist_msg)

    def circle_target(self, t, radius, center):
        angle = t * 0.1  # Angular velocity (radians per second)
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)

        dx = -radius * np.sin(angle)
        dy = radius * np.cos(angle)
        yaw = np.arctan2(dy, dx)

        return np.array([x, y, yaw])

    def circle_velocity_target(self, radius, v):
        angular_velocity = v / radius 
        vx_goal = v
        r_goal = angular_velocity
        return vx_goal, r_goal

    def calculate_lambda(self, vx, vy):
        phi = v_blend_min + 0.5 * (v_blend_max - v_blend_min)
        w = (2 * np.pi) / (v_blend_max - v_blend_min)
        lambda_val = 0.5 * (np.tanh(w * ((vx**2 + vy**2)**0.5 - phi)) + 1)
        return lambda_val

    def calculate_slip_angles(self, vx, vy, r, delta):
        epsilon = 1e-5
        vx_safe = max(vx, epsilon)
        alpha_f = -np.arctan((Lf * r + vy) / vx_safe) + delta
        alpha_r = np.arctan((Lr * r - vy) / vx_safe)
        if vx < epsilon:
            alpha_f = alpha_r = 0
        return alpha_f, alpha_r

    def calculate_tire_forces(self, alpha_f, alpha_r):
        Fyf = Df * np.sin(Cf * np.arctan(Bf * alpha_f))
        Fyr = Dr * np.sin(Cr * np.arctan(Br * alpha_r))
        return Fyf, Fyr

    def drift_model(self, state, control, dt):
        x, y, yaw, vx, vy, r, delta = state
        Fx, Delta_delta = control

        lam = self.calculate_lambda(vx, vy)
        alpha_f, alpha_r = self.calculate_slip_angles(vx, vy, r, delta)
        Fyf, Fyr = self.calculate_tire_forces(alpha_f, alpha_r)

        B = 0.1
        x_dot_dyn = vx * np.cos(yaw) - vy * np.sin(yaw)
        y_dot_dyn = vx * np.sin(yaw) + vy * np.cos(yaw)
        yaw_dot_dyn = r
        vx_dot_dyn = (1 / m) * (-B*vx + Fx - Fyf * np.sin(delta) + m * vy * r)
        vy_dot_dyn = (1 / m) * (-B*vy + Fyr + Fyf * np.cos(delta) - m * vx * r)
        r_dot_dyn = (1 / Iz) * (-B*r + Fyf * Lf * np.cos(delta) - Fyr * Lr)
        delta_dot_dyn = Delta_delta

        x_dot_kin = vx * np.cos(yaw) - vy * np.sin(yaw)
        y_dot_kin = vx * np.sin(yaw) + vy * np.cos(yaw)
        yaw_dot_kin = r
        vx_dot_kin = (-B*vx + Fx) / m
        vy_dot_kin = (Delta_delta * vx) * (Lr / (Lr + Lf))
        r_dot_kin = (Delta_delta * vx) * (1 / (Lr + Lf))
        delta_dot_kin = Delta_delta

        lam = 0.05
        x_dot = lam * x_dot_dyn + (1 - lam) * x_dot_kin
        y_dot = lam * y_dot_dyn + (1 - lam) * y_dot_kin
        yaw_dot = lam * yaw_dot_dyn + (1 - lam) * yaw_dot_kin
        vx_dot = lam * vx_dot_dyn + (1 - lam) * vx_dot_kin
        vy_dot = lam * vy_dot_dyn + (1 - lam) * vy_dot_kin
        r_dot = lam * r_dot_dyn + (1 - lam) * r_dot_kin
        delta_dot = lam * delta_dot_dyn + (1 - lam) * delta_dot_kin

        x += x_dot * dt
        y += y_dot * dt
        yaw += yaw_dot * dt  
        vx += vx_dot * dt
        vy += vy_dot * dt
        r += r_dot * dt
        delta += delta_dot * dt

        return np.array([x, y, yaw, vx, vy, r, delta])

    def mpc_cost(self, U, *args):
        N, state, target = args
        cost = 0.0

        vx_goal, r_goal = self.circle_velocity_target(circle_radius, 2.5)

        alpha_vx = 1.0
        alpha_r = 1.0

        for i in range(N):
            state = self.drift_model(state, U[2 * i:2 * i + 2], dt)
            x, y, yaw, vx, vy, r, delta = state

            w_ss = alpha_vx*(vx - vx_goal)**2 + alpha_r*(r - r_goal)**2
            cost += w_ss

        x_target, y_target, yaw_target = target
        position_error = (x - x_target)**2 + (y - y_target)**2
        diff_yaw = yaw - yaw_target
        heading_error = (np.arctan2(np.sin(diff_yaw), np.cos(diff_yaw)))**2

        J_park = position_error + heading_error + (vx**2) + (vy**2) + (r**2)
        return cost

def main(args=None):
    rclpy.init(args=args)
    drift_controller = DriftController()
    rclpy.spin(drift_controller)
    drift_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()