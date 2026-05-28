#!/usr/bin/env python3

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleStatus
from scipy.ndimage import distance_transform_edt

class ESDFOffboardController(Node):
    def __init__(self):
        super().__init__("esdf_offboard_controller")

        qos_px4 = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, depth=1)
        qos_rviz = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, depth=5)

        self.create_subscription(Path, "/planned_path", self.path_cb, qos_rviz)
        self.create_subscription(Odometry, "/odom", self.odom_cb, 10)
        self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status_v1", self.status_cb, qos_px4)
        
        # Subscribe to Nvblox Grid just for RViz Visualization
        self.create_subscription(OccupancyGrid, "/nvblox_node/static_occupancy_grid", self.map_cb, qos_rviz)
        self.inflated_map_pub = self.create_publisher(OccupancyGrid, "/inflated_map", qos_rviz)

        self.mode_pub = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", qos_px4)
        self.sp_pub = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos_px4)
        self.cmd_pub = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", qos_px4)

        self.state = "WARMUP"
        self.counter = 0

        self.position = np.zeros(3, dtype=float)
        self.yaw_enu = 0.0
        self.target_z = 2.0
        self.status = VehicleStatus()

        self.planned_path = np.zeros((0, 2), dtype=float)
        
        # Accumulator variables
        self.command_pos_enu = None
        self.filtered_step_xy = np.zeros(2, dtype=float)
        self.hover_pos = np.zeros(3, dtype=float)

        self.map_res = 0.10
        self.map_origin = np.zeros(2)
        self.esdf_m = None

        self.control_dt = 0.05
        self.timer = self.create_timer(self.control_dt, self.loop)

    def ts(self): return int(self.get_clock().now().nanoseconds / 1000)

    def odom_cb(self, msg):
        self.position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        q = msg.pose.pose.orientation
        self.yaw_enu = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def path_cb(self, msg):
        self.planned_path = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses], dtype=float)

    def status_cb(self, msg): self.status = msg

    def map_cb(self, msg: OccupancyGrid):
        # We only do this to publish the inflated map to RViz. 
        map_res = float(msg.info.resolution)
        self.map_origin = np.array([msg.info.origin.position.x, msg.info.origin.position.y])
        data = np.array(msg.data, dtype=np.int16).reshape((msg.info.height, msg.info.width))
        occupied = (data > 50).astype(np.uint8)
        self.esdf_m = distance_transform_edt(1 - occupied) * map_res

        inflated_msg = OccupancyGrid()
        inflated_msg.header = msg.header
        inflated_msg.info = msg.info
        inflated_data = np.where(self.esdf_m < 0.75, 100, 0)
        inflated_msg.data = inflated_data.astype(np.int8).flatten().tolist()
        self.inflated_map_pub.publish(inflated_msg)

    def command(self, cmd, p1=0.0, p2=0.0):
        msg = VehicleCommand(command=int(cmd), param1=float(p1), param2=float(p2), target_system=1, target_component=1, source_system=1, source_component=1, from_external=True, timestamp=self.ts())
        self.cmd_pub.publish(msg)

    def publish_setpoint_enu(self, pos_enu, yaw_enu):
        sp = TrajectorySetpoint()
        sp.position = [float(pos_enu[1]), float(pos_enu[0]), float(-pos_enu[2])]
        
        # Explicitly pass NaN for vel/acc so PX4 purely tracks the position
        sp.velocity = [float(np.nan), float(np.nan), float(np.nan)]
        sp.acceleration = [float(np.nan), float(np.nan), float(np.nan)]
        
        sp.yaw = float(math.atan2(math.cos(yaw_enu), math.sin(yaw_enu)))
        sp.timestamp = self.ts()
        self.sp_pub.publish(sp)

    def get_esdf_gradient(self, px, py):
        if self.esdf_m is None: return 999.0, np.zeros(2)
        
        gx = int(round((px - self.map_origin[0]) / self.map_res))
        gy = int(round((py - self.map_origin[1]) / self.map_res))
        
        if not (0 <= gy < self.esdf_m.shape[0] and 0 <= gx < self.esdf_m.shape[1]):
            return 0.0, np.zeros(2)

        dist = float(self.esdf_m[gy, gx])
        
        eps = max(self.map_res, 0.1)
        dx = (self.sample_esdf(px + eps, py) - self.sample_esdf(px - eps, py)) / (2.0 * eps)
        dy = (self.sample_esdf(px, py + eps) - self.sample_esdf(px, py - eps)) / (2.0 * eps)
        
        grad = np.array([dx, dy])
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-4: grad /= grad_norm
        
        return dist, grad

    def sample_esdf(self, x, y):
        gx = int(round((x - self.map_origin[0]) / self.map_res))
        gy = int(round((y - self.map_origin[1]) / self.map_res))
        if 0 <= gy < self.esdf_m.shape[0] and 0 <= gx < self.esdf_m.shape[1]: return self.esdf_m[gy, gx]
        return 0.0

    def loop(self):
        self.counter += 1
        msg = OffboardControlMode(position=True, timestamp=self.ts())
        self.mode_pub.publish(msg)

        if self.state == "WARMUP":
            if self.counter > 50: self.state = "OFFBOARD"
            self.publish_setpoint_enu(self.position, self.yaw_enu)

        elif self.state == "OFFBOARD":
            self.publish_setpoint_enu(self.position, self.yaw_enu)
            if self.counter % 5 == 0:
                self.command(176, 1.0, 6.0) # Offboard Mode
                self.command(400, 1.0)      # Arm
            if self.status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD and self.status.arming_state == VehicleStatus.ARMING_STATE_ARMED:
                self.state = "TAKEOFF"

        elif self.state == "TAKEOFF":
            takeoff_pos = np.array([self.position[0], self.position[1], self.target_z])
            self.publish_setpoint_enu(takeoff_pos, self.yaw_enu)
            if abs(self.position[2] - self.target_z) < 0.15:
                self.command_pos_enu = np.array([self.position[0], self.position[1], self.target_z])
                self.state = "HOVER"
                self.hover_pos = np.array([self.position[0], self.position[1], self.target_z])

        elif self.state == "HOVER":
            # If a new path is published and the final goal is significantly different, start moving again
            if len(self.planned_path) > 0:
                final_goal = self.planned_path[-1]
                px, py = self.position[0], self.position[1]
                dist_to_new_goal = float(np.linalg.norm(final_goal - np.array([px, py])))
                
                if dist_to_new_goal > 0.40:
                    # Transition to ALIGN_YAW instead of NAV to prevent sweeping turns
                    self.state = "ALIGN_YAW"
                    return
            
            # Rigidly hold this exact position. Ignores all APF forces and obstacles.
            self.publish_setpoint_enu(self.hover_pos, self.yaw_enu)

        elif self.state == "ALIGN_YAW":
            # Safely turn in place to face the path before moving forward
            if len(self.planned_path) == 0:
                self.state = "HOVER"
                return

            px, py = self.position[0], self.position[1]
            
            # Find a point slightly ahead on the path to look at
            distances = np.linalg.norm(self.planned_path - np.array([px, py]), axis=1)
            closest_idx = int(np.argmin(distances))
            
            target_idx = closest_idx
            for i in range(closest_idx, len(self.planned_path)):
                if np.linalg.norm(self.planned_path[i] - np.array([px, py])) > 0.5:
                    target_idx = i
                    break
            
            carrot = self.planned_path[target_idx]
            goal_vec = carrot - np.array([px, py])
            
            # Determine which way to face
            if np.linalg.norm(goal_vec) > 0.05:
                target_yaw = math.atan2(goal_vec[1], goal_vec[0])
            else:
                target_yaw = self.yaw_enu

            # Calculate the shortest rotational distance to the target yaw
            yaw_err = (target_yaw - self.yaw_enu + math.pi) % (2 * math.pi) - math.pi
            cmd_yaw = self.yaw_enu + np.clip(yaw_err, -2.0 * self.control_dt, 2.0 * self.control_dt)

            # Keep publishing the static hover position while rotating
            self.publish_setpoint_enu(self.hover_pos, cmd_yaw)

            # If pointed roughly at the target (within ~11 degrees), hand over to NAV
            if abs(yaw_err) < 0.2:
                self.state = "NAV"
                self.command_pos_enu = np.array([px, py, self.target_z], dtype=float)
                self.filtered_step_xy = np.zeros(2, dtype=float)

        elif self.state == "NAV":
            if len(self.planned_path) == 0:
                self.state = "HOVER"
                self.hover_pos = np.array([self.position[0], self.position[1], self.target_z])
                return

            px, py = self.position[0], self.position[1]
            final_goal = self.planned_path[-1]
            dist_to_final = float(np.linalg.norm(final_goal - np.array([px, py])))
            
            # Goal Arrival Logic: Switch completely to HOVER state
            if dist_to_final < 0.25:
                self.state = "HOVER"
                # Snap to the exact final goal coordinate
                self.hover_pos = np.array([final_goal[0], final_goal[1], self.target_z])
                self.publish_setpoint_enu(self.hover_pos, self.yaw_enu)
                return
            
            # Find closest point on B-Spline
            distances = np.linalg.norm(self.planned_path - np.array([px, py]), axis=1)
            closest_idx = int(np.argmin(distances))
            
            # Look ahead 0.5m
            target_idx = closest_idx
            for i in range(closest_idx, len(self.planned_path)):
                if np.linalg.norm(self.planned_path[i] - np.array([px, py])) > 0.5:
                    target_idx = i
                    break
            
            carrot = self.planned_path[target_idx]
            
            # 1. Pure Path Tracking
            goal_vec = carrot - np.array([px, py])
            tracking_dir = goal_vec / max(np.linalg.norm(goal_vec), 1e-4)
            
            # Slightly increased top speed to 0.9 m/s
            max_speed = 0.90 
            current_speed = max_speed
            
            if dist_to_final < 1.0:
                current_speed = max_speed * (dist_to_final / 1.0)
                current_speed = max(current_speed, 0.15) 
            
            desired_vel = tracking_dir * current_speed

            # 2. Classic Linear APF
            dist_to_obs, grad = self.get_esdf_gradient(px, py)
            inflation_radius = 0.75
            
            if dist_to_obs < inflation_radius:
                repulsive_strength = 2.5 * (inflation_radius - dist_to_obs)
                desired_vel += grad * repulsive_strength

            vel_norm = np.linalg.norm(desired_vel)
            if vel_norm > max_speed + 0.3:
                desired_vel = (desired_vel / vel_norm) * (max_speed + 0.3)

            # 3. Constant Step Filtering
            desired_step_xy = desired_vel * self.control_dt
            self.filtered_step_xy = 0.75 * self.filtered_step_xy + 0.25 * desired_step_xy
            
            if self.command_pos_enu is None:
                self.command_pos_enu = np.array([px, py, self.target_z], dtype=float)
                
            self.command_pos_enu[0] += float(self.filtered_step_xy[0])
            self.command_pos_enu[1] += float(self.filtered_step_xy[1])
            self.command_pos_enu[2] = self.target_z

            # Anti-windup leash
            carrot_offset_vec = self.command_pos_enu[:2] - np.array([px, py])
            carrot_dist = np.linalg.norm(carrot_offset_vec)
            if carrot_dist > 0.4:  
                self.command_pos_enu[:2] = np.array([px, py]) + (carrot_offset_vec / carrot_dist) * 0.4
            
            if np.linalg.norm(self.filtered_step_xy) > (0.01 * self.control_dt):
                target_yaw = math.atan2(self.filtered_step_xy[1], self.filtered_step_xy[0])
                yaw_err = (target_yaw - self.yaw_enu + math.pi) % (2 * math.pi) - math.pi
                cmd_yaw = self.yaw_enu + np.clip(yaw_err, -2.0 * self.control_dt, 2.0 * self.control_dt)
            else:
                cmd_yaw = self.yaw_enu

            self.publish_setpoint_enu(self.command_pos_enu, cmd_yaw)

def main():
    rclpy.init()
    try: rclpy.spin(ESDFOffboardController())
    finally: rclpy.shutdown()

if __name__ == "__main__": main()
