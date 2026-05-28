#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleOdometry
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

class PX4Bridge(Node):
    def __init__(self):
        super().__init__("px4_bridge")
        self.set_parameters([rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, True)])

        qos_px4 = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=10)
        qos_ros = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=10)

        self.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry", self.odom_cb, qos_px4)
        self.odom_pub = self.create_publisher(Odometry, "/odom", qos_ros)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_broadcaster = StaticTransformBroadcaster(self)
        
        self.timer = self.create_timer(1.0, self.publish_static_transforms)

    def publish_static_transforms(self):
        stamp = self.get_clock().now().to_msg()
        if stamp.sec == 0 and stamp.nanosec == 0: return

        transforms = []

        t_map = TransformStamped()
        t_map.header.stamp = stamp
        t_map.header.frame_id = "map"
        t_map.child_frame_id = "odom"
        t_map.transform.rotation.w = 1.0
        transforms.append(t_map)

        t_cam = TransformStamped()
        t_cam.header.stamp = stamp
        t_cam.header.frame_id = "base_link"
        t_cam.child_frame_id = "camera_link"
        t_cam.transform.translation.x = 0.10
        t_cam.transform.translation.z = 0.05
        
        rot = R.from_euler('xyz', [-np.pi/2, 0, -np.pi/2]).as_quat()
        t_cam.transform.rotation.x = float(rot[0])
        t_cam.transform.rotation.y = float(rot[1])
        t_cam.transform.rotation.z = float(rot[2])
        t_cam.transform.rotation.w = float(rot[3])
        transforms.append(t_cam)

        self.static_broadcaster.sendTransform(transforms)

    def odom_cb(self, msg):
        stamp = self.get_clock().now().to_msg()
        if stamp.sec == 0 and stamp.nanosec == 0: return

        x, y, z = msg.position[1], msg.position[0], -msg.position[2]
        vx, vy, vz = msg.velocity[1], msg.velocity[0], -msg.velocity[2]

        q_px4 = np.array([msg.q[1], msg.q[2], msg.q[3], msg.q[0]])
        rot_px4 = R.from_quat(q_px4)

        ned_to_enu = R.from_matrix([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        frd_to_flu = R.from_matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        q = (ned_to_enu * rot_px4 * frd_to_flu).as_quat()

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = float(x), float(y), float(z)
        odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = float(vx), float(vy), float(vz)
        self.odom_pub.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = "odom"
        tf.child_frame_id = "base_link"
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = float(x), float(y), float(z)
        tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        self.tf_broadcaster.sendTransform(tf)

def main():
    rclpy.init()
    node = PX4Bridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
