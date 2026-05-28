#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import math
import heapq
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import splprep, splev
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class StrictAstarNavigator(Node):
    def __init__(self):
        super().__init__("smooth_navigator")
        
        qos_standard = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, depth=5)
        qos_rviz = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, depth=5)

        # Nvblox Occupancy Grid
        self.create_subscription(OccupancyGrid, "/nvblox_node/static_occupancy_grid", self.map_cb, qos_standard)
        self.create_subscription(Odometry, "/odom", self.odom_cb, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self.goal_cb, qos_standard)

        self.path_pub = self.create_publisher(Path, "/planned_path", qos_rviz)

        self.res = 0.10
        self.grid_width = 0
        self.grid_height = 0
        self.origin_x = 0.0
        self.origin_y = 0.0

        self.raw_grid = None
        self.map_received = False
        
        # CRITICAL FIX: Matched to the Offboard Controller's APF radius
        self.safety_radius_m = 0.75 

        self.global_goal = None
        self.position = np.zeros(3, dtype=float)

        self.timer = self.create_timer(0.5, self.planning_loop) # Run at 2Hz

        self.get_logger().info("Strict A* Navigator (Center-line Preference) Active")

    def goal_cb(self, msg):
        self.global_goal = np.array([msg.pose.position.x, msg.pose.position.y], dtype=float)
        self.get_logger().info(f"Goal set: ({self.global_goal[0]:.2f}, {self.global_goal[1]:.2f})")

    def odom_cb(self, msg):
        self.position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z], dtype=float)

    def map_cb(self, msg: OccupancyGrid):
        self.map_received = True
        self.res = float(msg.info.resolution)
        self.grid_width = int(msg.info.width)
        self.grid_height = int(msg.info.height)
        self.origin_x = float(msg.info.origin.position.x)
        self.origin_y = float(msg.info.origin.position.y)

        data = np.array(msg.data, dtype=np.int16).reshape((self.grid_height, self.grid_width))
        self.raw_grid = (data > 50).astype(np.uint8)

    def world_to_grid(self, x, y):
        gx = int(np.clip((x - self.origin_x) / self.res, 0, self.grid_width - 1))
        gy = int(np.clip((y - self.origin_y) / self.res, 0, self.grid_height - 1))
        return gx, gy

    def grid_to_world(self, gx, gy):
        return self.origin_x + gx * self.res, self.origin_y + gy * self.res

    def astar(self, start_idx, goal_idx, inflated_grid, clearance_m):
        def h(a, b): return math.hypot(a[0] - b[0], a[1] - b[1])
        
        open_set = []
        heapq.heappush(open_set, (0.0, start_idx))
        came_from = {}
        g_score = {start_idx: 0.0}
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_idx:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            cx, cy = current
            for dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.grid_width and 0 <= ny < self.grid_height): continue
                if inflated_grid[ny, nx] != 0: continue
                
                # Prevent diagonal corner cutting
                if dx != 0 and dy != 0 and (inflated_grid[cy, nx] != 0 or inflated_grid[ny, cx] != 0): continue

                # CRITICAL FIX: Center-line preference to stop path flickering
                dist_to_obs = clearance_m[ny, nx]
                safety_penalty = 0.0
                if dist_to_obs < 1.5:
                    safety_penalty = 2.0 * (1.5 - dist_to_obs)

                tentative_g = g_score[current] + (1.414 if dx != 0 and dy != 0 else 1.0) + safety_penalty
                
                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    came_from[(nx, ny)] = current
                    g_score[(nx, ny)] = tentative_g
                    heapq.heappush(open_set, (tentative_g + h((nx, ny), goal_idx), (nx, ny)))
        return []

    def densify_path(self, path, step=0.1):
        """ Adds points along straight lines so B-spline can't cut corners """
        if len(path) < 2: return path
        dense = []
        for i in range(len(path)-1):
            p1 = path[i]
            p2 = path[i+1]
            dist = np.linalg.norm(p2 - p1)
            num_points = max(1, int(dist / step))
            for j in range(num_points):
                dense.append(p1 + (p2 - p1) * (j / num_points))
        dense.append(path[-1])
        return dense

    def generate_bspline_trajectory(self, path_world):
        if len(path_world) < 3:
            return path_world 
            
        # 1. Densify to lock the spline to the safe A* lines
        dense_path = self.densify_path(path_world, step=0.1)
        
        # 2. Clean duplicates to prevent scipy crash
        clean_path = [dense_path[0]]
        for p in dense_path[1:]:
            if np.linalg.norm(p - clean_path[-1]) > 0.01:
                clean_path.append(p)
                
        if len(clean_path) < 3:
            return clean_path
            
        x = [p[0] for p in clean_path]
        y = [p[1] for p in clean_path]

        # 3. Smoothing factor reduced drastically (s=0.2) so it can't pull out of bounds
        tck, u = splprep([x, y], s=0.2, k=min(3, len(clean_path)-1))
        
        u_fine = np.linspace(0, 1, max(50, len(clean_path)))
        x_fine, y_fine = splev(u_fine, tck)
        
        return [np.array([x_fine[i], y_fine[i]]) for i in range(len(x_fine))]

    def prune_path(self, path):
        if len(path) <= 2: return path
        pruned = [path[0]]
        for i in range(1, len(path) - 1):
            dx1, dy1 = path[i][0] - path[i-1][0], path[i][1] - path[i-1][1]
            dx2, dy2 = path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]
            if dx1*dy2 != dx2*dy1:
                pruned.append(path[i])
        pruned.append(path[-1])
        return pruned

    def planning_loop(self):
        if not self.map_received or self.raw_grid is None or self.global_goal is None: return

        # Get ESDF via distance transform
        dist_cells = distance_transform_edt(1 - self.raw_grid)
        clearance_m = dist_cells * self.res
        inflated_grid = (clearance_m <= self.safety_radius_m).astype(np.uint8)

        sgx, sgy = self.world_to_grid(self.position[0], self.position[1])
        ggx, ggy = self.world_to_grid(self.global_goal[0], self.global_goal[1])

        # Pass clearance to A* for the center-line penalty
        grid_path = self.astar((sgx, sgy), (ggx, ggy), inflated_grid, clearance_m)
        
        if not grid_path:
            self.get_logger().warning("No safe path found!")
            return

        # 1. Convert to world coordinates
        world_path = [np.array(self.grid_to_world(x, y)) for x, y in grid_path]
        
        # 2. Prune redundant straight lines
        pruned_path = self.prune_path(world_path)
        
        # 3. Apply safe B-Spline Smoothing
        smooth_trajectory = self.generate_bspline_trajectory(pruned_path)

        # Publish smooth path
        planned = Path()
        planned.header.stamp = self.get_clock().now().to_msg()
        planned.header.frame_id = "odom"
        for p in smooth_trajectory:
            pose = PoseStamped()
            pose.pose.position.x = float(p[0])
            pose.pose.position.y = float(p[1])
            pose.pose.position.z = 2.0
            planned.poses.append(pose)
            
        self.path_pub.publish(planned)

def main():
    rclpy.init()
    node = StrictAstarNavigator()
    try: rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__": main()
