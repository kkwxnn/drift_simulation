#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Use a non-blocking matplotlib backend
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for non-blocking plots

class DebugPlotter(Node):
    def __init__(self):
        super().__init__('debug_plotter')
        # Subscribe to debug data
        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/debug_data',
            self.debug_callback,
            10)
        
        # Initialize empty lists to store data
        self.time_data = []
        self.x_data = []  # X position of the car
        self.y_data = []  # Y position of the car
        self.yaw_data = []  # Yaw (heading) of the car
        self.delta_data = []  # Steering angle
        self.r_data = []  # Yaw rate
        self.vx_data = []  # Longitudinal velocity
        self.vy_data = []  # Lateral velocity
        self.fx_data = []  # Force input (Fx)
        self.delta_delta_data = []  # Steering angle change (Delta_delta)
        self.cost_data = []  # MPC cost

        # Set up plots
        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 12))
        self.fig.suptitle("Drift Controller Debug Plots")

        # Set up subplots
        self.axs[0].set_title("Trajectory (CoG and Heading)")
        self.axs[0].set_xlabel("X Position (m)")
        self.axs[0].set_ylabel("Y Position (m)")

        self.axs[1].set_title("MPC Cost")
        self.axs[1].set_xlabel("Time (s)")
        self.axs[1].set_ylabel("Cost")

        self.axs[2].set_title("Control Inputs (Fx, Delta_delta)")
        self.axs[2].set_xlabel("Time (s)")
        self.axs[2].set_ylabel("Control Input")

        # Enable grid for all subplots
        for ax in self.axs:
            ax.grid(True)

        # Adjust layout
        plt.tight_layout()

        # Start the animation
        self.ani = animation.FuncAnimation(self.fig, self.update_plots, interval=100)
        plt.show(block=False)  # Use non-blocking show

    def debug_callback(self, msg):
        """Callback for debug data."""
        # Append new data
        t = len(self.time_data) * 0.1  # Simulated time step
        self.time_data.append(t)
        self.x_data.append(msg.data[0])  # X position (from state)
        self.y_data.append(msg.data[1])  # Y position (from state)
        self.yaw_data.append(msg.data[2])  # Yaw (heading)
        self.delta_data.append(msg.data[3])  # Steering angle
        self.r_data.append(msg.data[4])  # Yaw rate
        self.vx_data.append(msg.data[5])  # Longitudinal velocity
        self.vy_data.append(msg.data[6])  # Lateral velocity
        self.fx_data.append(msg.data[7])  # Force input (Fx)
        self.delta_delta_data.append(msg.data[8])  # Steering angle change (Delta_delta)
        self.cost_data.append(msg.data[9])  # MPC cost

    def update_plots(self, _):
        """Update the plots with new data."""
        # Clear previous plots
        for ax in self.axs:
            ax.clear()

        # Plot trajectory (CoG and heading)
        self.axs[0].plot(self.x_data, self.y_data, label="CoG Trajectory", color='blue')
        # Plot heading direction as arrows
        for i in range(0, len(self.x_data), 10):  # Plot every 10th step to avoid clutter
            dx = 0.5 * np.cos(self.yaw_data[i])  # Scale arrows for better visualization
            dy = 0.5 * np.sin(self.yaw_data[i])
            self.axs[0].arrow(self.x_data[i], self.y_data[i], dx, dy, head_width=0.2, head_length=0.3, fc='red', ec='red')
        self.axs[0].set_title("Trajectory (CoG and Heading)")
        self.axs[0].legend()

        # Plot MPC cost
        self.axs[1].plot(self.time_data, self.cost_data, label="Cost", color='green')
        self.axs[1].set_title("MPC Cost")
        self.axs[1].legend()

        # Plot control inputs (Fx, Delta_delta)
        self.axs[2].plot(self.time_data, self.fx_data, label="Fx", color='orange')
        self.axs[2].plot(self.time_data, self.delta_delta_data, label="Delta_delta", color='purple')
        self.axs[2].set_title("Control Inputs (Fx, Delta_delta)")
        self.axs[2].legend()

        # Redraw the figure
        plt.draw()
        plt.pause(0.01)

    def shutdown(self):
        """Cleanup on shutdown."""
        plt.close(self.fig)

def main(args=None):
    rclpy.init(args=args)
    debug_plotter = DebugPlotter()
    
    try:
        rclpy.spin(debug_plotter)
    except KeyboardInterrupt:
        pass  # Handle Ctrl+C gracefully
    finally:
        # Cleanup
        debug_plotter.shutdown()
        debug_plotter.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()