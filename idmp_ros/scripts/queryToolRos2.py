#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ROS2 Query Tool for IDMP (ported from ROS1 version)
# Author: Lan Wu, 2025

import numpy as np
if not hasattr(np, 'float'):
    np.float = float

# import matplotlib
# matplotlib.use("Qt5Agg")
# import matplotlib.pyplot as plt

import tf_transformations as transformations

import math

import rclpy
from rclpy.node import Node

from builtin_interfaces.msg import Time as TimeMsg
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped, Point, Pose, Quaternion
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray, InteractiveMarker, InteractiveMarkerControl
from sensor_msgs_py import point_cloud2 as pc2

import tf2_ros
from tf2_ros import TransformException, TransformBroadcaster
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from idmp_interfaces.srv import GetDistanceGradient

def pose_to_numpy(msg):
    return np.dot(
        transformations.translation_matrix(np.array([msg.position.x, msg.position.y, msg.position.z])),
        transformations.quaternion_matrix(np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]))
    )

def make_grid_xy(nx=21, ny=21, res=0.05, z=0.0):
    xs = (np.arange(nx) - (nx - 1) / 2.0) * res
    ys = (np.arange(ny) - (ny - 1) / 2.0) * res
    xv, yv = np.meshgrid(xs, ys, indexing='xy')
    pts = np.stack([xv.reshape(-1), yv.reshape(-1), np.full(xv.size, z)], axis=1).astype(np.float32)
    return pts  # [N,3]

def pose_matrix_from_tf(tf: TransformStamped) -> np.ndarray:
    t = tf.transform.translation
    q = tf.transform.rotation
    tx, ty, tz = float(t.x), float(t.y), float(t.z)
    qw, qx, qy, qz = float(q.w), float(q.x), float(q.y), float(q.z)
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)]
    ], dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def apply_transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    N = pts.shape[0]
    homo = np.concatenate([pts, np.ones((N, 1), dtype=np.float32)], axis=1)
    out = (T @ homo.T).T[:, :3]
    return out.astype(np.float32)


def make_pointcloud2(points_xyz: np.ndarray, intensities: np.ndarray,
                     frame_id: str, stamp: TimeMsg) -> PointCloud2:
    header = Header()
    header.frame_id = frame_id
    header.stamp = stamp
    fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    data = np.c_[points_xyz.astype(np.float32), intensities.astype(np.float32)]
    return pc2.create_cloud(header, fields, data.tolist())


def make_arrow_marker(idx: int, p0: np.ndarray, p1: np.ndarray, rgba=(0.2, 0.8, 0.2, 1.0),
                      frame_id="base", stamp: TimeMsg = None,
                      shaft_d=0.02, head_d=0.04, head_l=0.06) -> Marker:
    m = Marker()
    m.header.frame_id = frame_id
    if stamp is not None:
        m.header.stamp = stamp
    m.ns = "query_grad"
    m.id = idx
    m.type = Marker.ARROW
    m.action = Marker.ADD
    m.scale.x = float(shaft_d)
    m.scale.y = float(head_d)
    m.scale.z = float(head_l)
    m.color.r, m.color.g, m.color.b, m.color.a = rgba
    ps = Point(x=float(p0[0]), y=float(p0[1]), z=float(p0[2]))
    pe = Point(x=float(p1[0]), y=float(p1[1]), z=float(p1[2]))
    m.points = [ps, pe]
    m.lifetime.sec = 0
    m.frame_locked = False
    return m


class QueryToolRos2(Node):
    def __init__(self):
        super().__init__("query_tool_ros2")

        self.declare_parameter("world_frame", "base")
        self.declare_parameter("tool_frame", "camera_link")
        self.declare_parameter("query_rate", 5.0)
        self.declare_parameter("grid_nx", 42)
        self.declare_parameter("grid_ny", 42)
        self.declare_parameter("grid_res", 0.05)
        self.declare_parameter("grid_z", 0.0)
        self.declare_parameter("arrow_stride", 2)
        self.declare_parameter("grad_scale", 0.15)
        self.declare_parameter("distance_clip", 5.0)

        self.declare_parameter("enable_interactive_marker", True)
        self.declare_parameter("marker_name", "query_tool")
        self.declare_parameter("init_xyz", [0.0, 0.0, 0.5])
        self.declare_parameter("init_rpy", [0.0, 0.0, 0.0])  # radians

        # motion switch：wave / hMove / vMove / heightfak
        self.declare_parameter("wave", False)
        self.declare_parameter("hMove", False)
        self.declare_parameter("vMove", False)

        # load params
        self.world_frame = self.get_parameter("world_frame").get_parameter_value().string_value
        self.tool_frame  = self.get_parameter("tool_frame").get_parameter_value().string_value
        self.query_rate  = self.get_parameter("query_rate").get_parameter_value().double_value
        self.grid_nx     = self.get_parameter("grid_nx").get_parameter_value().integer_value
        self.grid_ny     = self.get_parameter("grid_ny").get_parameter_value().integer_value
        self.grid_res    = self.get_parameter("grid_res").get_parameter_value().double_value
        self.grid_z      = self.get_parameter("grid_z").get_parameter_value().double_value
        self.arrow_stride = self.get_parameter("arrow_stride").get_parameter_value().integer_value
        self.grad_scale  = self.get_parameter("grad_scale").get_parameter_value().double_value
        self.distance_clip = self.get_parameter("distance_clip").get_parameter_value().double_value

        self.enable_im   = self.get_parameter("enable_interactive_marker").get_parameter_value().bool_value
        self.marker_name = self.get_parameter("marker_name").get_parameter_value().string_value
        xyz = list(self.get_parameter("init_xyz").get_parameter_value().double_array_value)
        rpy = list(self.get_parameter("init_rpy").get_parameter_value().double_array_value)

        self.wave = self.get_parameter("wave").get_parameter_value().bool_value
        self.hMove = self.get_parameter("hMove").get_parameter_value().bool_value
        self.vMove = self.get_parameter("vMove").get_parameter_value().bool_value

        # === variable：wave / heightfak ===
        self.waveStep = 0.0
        self.heightfak = -0.7  # initial values，[-0.7, 0.5] loop them

        self.query_grid_local = make_grid_xy(self.grid_nx, self.grid_ny, self.grid_res, self.grid_z)  # [N,3]
        self.N_pts = self.query_grid_local.shape[0]

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tfb = TransformBroadcaster(self)

        self.pcl_pub = self.create_publisher(PointCloud2, "distfield", 1)
        self.marker_pub = self.create_publisher(MarkerArray, "grad_array", 1)

        # Service Client
        self.client = self.create_client(GetDistanceGradient, "query_dist_field")
        self.pending_future = None

        self.server = None
        self.server = InteractiveMarkerServer(self, 'query_tool_marker_server')
        
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "base"
        int_marker.name = "dist_field_center"
        int_marker.description = "Distance and Gradient to Collision for Safe Human-Robot Interactions"

        # create a grey box marker
        box_marker = Marker()
        box_marker.type = Marker.CUBE
        box_marker.scale.x = 0.45
        box_marker.scale.y = 0.45
        box_marker.scale.z = 0.45
        box_marker.color.r = 0.0
        box_marker.color.g = 0.5
        box_marker.color.b = 0.5
        box_marker.color.a = 0.2

        # create a non-interactive control which contains the box
        box_control = InteractiveMarkerControl()
        box_control.always_visible = True
        box_control.interaction_mode = InteractiveMarkerControl.MOVE_ROTATE_3D
        box_control.markers.append( box_marker )
        int_marker.controls.append( box_control )

        pose = pose_to_numpy(box_marker.pose)
        new_data = True

        self.server.insert(int_marker)
        self.server.setCallback(int_marker.name, self._on_im_feedback)

        self.server.applyChanges()

        # save the IM's pose as（world←marker）
        self.pose_T_w_t = np.eye(4, dtype=np.float32)             # current T_world_tool
        self.pose_T_w_t[:] = pose_to_numpy(Pose())                # default: indentity matrix
        self.new_data = False     
            
        # main： check with timer
        self.create_timer(1.0 / max(1e-3, self.query_rate), self._on_timer)

        self.get_logger().info(
            f"QueryToolRos2 ready. world={self.world_frame}, tool={self.tool_frame}, "
            f"N={self.N_pts}, rate={self.query_rate}Hz, wave={self.wave}, hMove={self.hMove}, vMove={self.vMove}"
        )

    def _init_pose_from_xyz_rpy(self, xyz, rpy) -> Pose:
        p = Pose()
        p.position.x, p.position.y, p.position.z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        q = self._rpy_to_quat(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        p.orientation = q
        return p

    def _on_im_feedback(self, feedback):
        # feedback from the interactive marker：update T_world_tool
        self.pose_T_w_t = pose_to_numpy(feedback.pose).astype(np.float32)
        self.new_data = True

    # === main loop：query grid -> motion query -> TF -> service -> visualisation ===
    def _on_timer(self):
        if self.pending_future is not None and not self.pending_future.done():
            return

        # 1) local query grid + motion query
        pts_local = self.query_grid_local.copy()
        if self.hMove or self.vMove:
            self.heightfak += 0.02
            if self.heightfak > 0.5:
                self.heightfak = -0.7
            pts_local[:, 2] += self.heightfak
        if self.wave:
            self.waveStep += 0.2
            pts_local[:, 2] += 0.05 * float(np.sin(self.waveStep))
        if self.hMove:
            pts_local[:, 0] += 0.05 * float(np.sin(self.waveStep * 0.7))
        if self.vMove:
            pts_local[:, 1] += 0.05 * float(np.cos(self.waveStep * 0.9))

        # 2) world pose transformation：use IM's pose；otherwise, use TF pose
        if self.enable_im:
            T_w_t = self.pose_T_w_t
        else:
            try:
                tf = self.tf_buffer.lookup_transform(self.world_frame, self.tool_frame, rclpy.time.Time())
                T_w_t = pose_matrix_from_tf(tf)
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed: {ex}")
                return

        # 3) send the service
        pts_world = apply_transform(T_w_t, pts_local)
        if not self.client.service_is_ready():
            self.client.wait_for_service(timeout_sec=0.0)
            return
        req = GetDistanceGradient.Request()
        req.points = pts_world.reshape(-1).astype(np.float32).tolist()
        self.pending_future = self.client.call_async(req)
        self.pending_future.add_done_callback(lambda fut, p=pts_world: self._on_query_done(fut, p))


    def _on_query_done(self, future, pts_world: np.ndarray):
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().warn(f"Service call failed: {e}")
            self.pending_future = None
            return

        d = np.array(res.distances, dtype=np.float32)
        g = np.array(res.gradients, dtype=np.float32).reshape((-1, 3)) if len(res.gradients) > 0 else np.zeros((pts_world.shape[0], 3), dtype=np.float32)
        if d.shape[0] != pts_world.shape[0]:
            self.get_logger().warn("Response size mismatch with query points.")
            self.pending_future = None
            return

        if self.distance_clip > 0:
            mask = d < float(self.distance_clip)
        else:
            mask = np.ones_like(d, dtype=bool)

        pts_vis = pts_world[mask]
        d_vis   = d[mask]
        g_vis   = g[mask]

        stamp = self.get_clock().now().to_msg()

        # pointcloud（intesity=distance），topic：distfield
        pcl_msg = make_pointcloud2(pts_vis, d_vis, self.world_frame, stamp)
        self.pcl_pub.publish(pcl_msg)

        # gradient arrow（topic：grad_array）
        marray = MarkerArray()
        stride = max(1, int(self.arrow_stride))
        for idx in range(0, pts_vis.shape[0], stride):
            p0 = pts_vis[idx]
            p1 = p0 + g_vis[idx] * float(self.grad_scale)
            m = make_arrow_marker(idx, p0, p1, rgba=(0.2, 0.8, 0.2, 1.0),
                                  frame_id=self.world_frame, stamp=stamp,
                                  shaft_d=0.02, head_d=0.04, head_l=0.06)
            marray.markers.append(m)
        self.marker_pub.publish(marray)

        self.pending_future = None

    # === tool function ===
    @staticmethod
    def _rpy_to_quat(roll, pitch, yaw) -> Quaternion:
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        qw = cr * cp * cy + sr * sp * sy
        q = Quaternion()
        q.x, q.y, q.z, q.w = qx, qy, qz, qw
        return q


def main():
    rclpy.init()
    node = QueryToolRos2()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
