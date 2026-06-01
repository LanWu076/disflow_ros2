#!/usr/bin/env python3

import time
import random
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
from tf2_ros import TransformException, TransformBroadcaster
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from idmp_interfaces.srv import GetDistanceGradient

class SoccerDemoNode(Node):
    def __init__(self):
        super().__init__('soccer')

        # -----------------------------
        # parameters / states
        # -----------------------------
        self.shot_threshold = 0.6
        self.old_dist = [0.0] * 11

        self.originx = 0.0
        self.originy = 0.0
        self.originz = 1.0

        self.textx = 4.0
        self.texty = 0.0
        self.textz = 2.0

        self.ball_pose = np.array([self.originx, self.originy, self.originz], dtype=float)
        self.ball_vel = 0.0
        self.ball_vel_max = 5.0
        self.ball_vel_damping = 0.95
        self.ball_direction = np.array([0.0, 0.0, 0.0], dtype=float)

        self.field_size = [4.0, 2.0, 2.0]
        self.goal_pole_pos = 1.0
        self.update_rate = 0.05

        self.block_until_time = None
        self.pending_future = None

        # -----------------------------
        # publishers
        # -----------------------------
        self.goal_pub = self.create_publisher(MarkerArray, 'goal', 10)
        self.ball_pub = self.create_publisher(Marker, 'Ball', 10)
        self.goal_effect_pub = self.create_publisher(Marker, 'goal_effect', 10)
        self.goal_text_pub = self.create_publisher(Marker, 'goal_text', 10)
        self.goal_fireworks_pub = self.create_publisher(MarkerArray, 'goal_fireworks', 10)

        # -----------------------------
        # service client
        # -----------------------------
        self.query_client = self.create_client(GetDistanceGradient, 'query_dist_field')
        while not self.query_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for service query_dist_field ...')

        # -----------------------------
        # init visualization
        # -----------------------------
        self.setup_goal()
        for _ in range(3):
            self.setup_goal()
            time.sleep(0.1)

        self.update_vis()

        # -----------------------------
        # timer
        # -----------------------------
        self.timer = self.create_timer(self.update_rate, self.timer_callback)

    # =========================================================
    # Helpers
    # =========================================================
    def _now(self):
        return self.get_clock().now().to_msg()

    def _duration_msg(self, sec: float):
        return Duration(seconds=sec).to_msg()

    # =========================================================
    # Marker creation
    # =========================================================
    def create_pole(self, x, y, z, y_scale, z_scale, idnum):
        m = Marker()
        m.header.frame_id = 'base'
        m.header.stamp = self._now()
        m.ns = "goal"
        m.id = idnum
        m.type = Marker.CUBE
        m.action = Marker.ADD

        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = float(z)
        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0

        m.scale.x = 0.05
        m.scale.y = float(y_scale)
        m.scale.z = float(z_scale)

        m.color.r = 0.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 1.0
        return m

    def create_ball(self, x, y, z, idnum):
        m = Marker()
        m.header.frame_id = 'base'
        m.header.stamp = self._now()
        m.ns = "ball"
        m.id = idnum
        m.type = Marker.SPHERE
        m.action = Marker.ADD

        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = float(z)
        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0

        m.scale.x = 0.5
        m.scale.y = 0.5
        m.scale.z = 0.5

        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 1.0
        return m

    def create_goal_text(self, x, y, z, idnum, msg):
        m = Marker()
        m.header.frame_id = 'base'
        m.header.stamp = self._now()
        m.ns = "goal_text"
        m.id = idnum
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD

        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = float(z + 1.0)
        m.pose.orientation.w = 1.0

        m.scale.z = 1.0

        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 1.0

        m.text = msg
        m.lifetime = self._duration_msg(3.0)
        return m

    def create_goal_effect(self, x, y, z, idnum, scale=1.0):
        m = Marker()
        m.header.frame_id = 'base'
        m.header.stamp = self._now()
        m.ns = "goal_effect"
        m.id = idnum
        m.type = Marker.SPHERE
        m.action = Marker.ADD

        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = float(z)
        m.pose.orientation.w = 1.0

        m.scale.x = float(scale)
        m.scale.y = float(scale)
        m.scale.z = float(scale)

        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 0.6

        m.lifetime = self._duration_msg(3.0)
        return m

    def create_goal_fireworks(self, x, y, z, count=20):
        arr = MarkerArray()

        for i in range(count):
            m = Marker()
            m.header.frame_id = 'base'
            m.header.stamp = self._now()
            m.ns = "goal_fireworks"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD

            offset = np.random.uniform(-1.0, 1.0, 3)
            offset[2] = abs(offset[2]) + 0.2

            m.pose.position.x = float(x + offset[0])
            m.pose.position.y = float(y + offset[1])
            m.pose.position.z = float(z + offset[2])
            m.pose.orientation.w = 1.0

            s = random.uniform(0.1, 0.3)
            m.scale.x = s
            m.scale.y = s
            m.scale.z = s

            m.color.r = random.random()
            m.color.g = random.random()
            m.color.b = random.random()
            m.color.a = 0.9

            m.lifetime = self._duration_msg(3.0)
            arr.markers.append(m)

        return arr

    # =========================================================
    # Game setup / reset
    # =========================================================
    def setup_goal(self):
        arr = MarkerArray()
        arr.markers.append(
            self.create_pole(
                self.field_size[0],
                self.goal_pole_pos,
                self.field_size[2] / 2.0,
                0.05,
                self.field_size[2],
                1
            )
        )
        arr.markers.append(
            self.create_pole(
                self.field_size[0],
                -self.goal_pole_pos,
                self.field_size[2] / 2.0,
                0.05,
                self.field_size[2],
                2
            )
        )
        arr.markers.append(
            self.create_pole(
                self.field_size[0],
                0.0,
                self.field_size[2],
                self.goal_pole_pos * 2.0,
                0.05,
                3
            )
        )
        self.goal_pub.publish(arr)

    def reset_ball(self):
        self.ball_pose = np.array([self.originx, self.originy, self.originz], dtype=float)
        self.ball_vel = 0.0
        self.ball_direction = np.array([0.0, 0.0, 0.0], dtype=float)
        #self.old_dist = [0.0] * 11
        self.update_vis()

    # =========================================================
    # Service query
    # =========================================================
    def call_query_service(self):
        if self.pending_future is not None:
            return

        req = GetDistanceGradient.Request()
        req.points = [
            float(self.ball_pose[0]),
            float(self.ball_pose[1]),
            float(self.ball_pose[2])
        ]

        self.pending_future = self.query_client.call_async(req)
        self.pending_future.add_done_callback(self.handle_query_response)

    def handle_query_response(self, future):
        self.pending_future = None

        try:
            response = future.result()
        except Exception as e:
            self.get_logger().warning(f'Service call failed: {e}')
            return

        if response is None:
            self.get_logger().warning('Service returned None.')
            return

        if len(response.in_bounds) > 0 and not response.in_bounds[0]:
            self.get_logger().warning('Ball query point out of bounds.')
            return

        if len(response.distances) < 1:
            self.get_logger().warning('No distance returned.')
            return

        if len(response.gradients) < 3:
            self.get_logger().warning('No valid gradient returned.')
            return

        dist = float(response.distances[0])
        grad = np.array(response.gradients[:3], dtype=float)

        self.calc_ball_params(dist, grad)
        self.calc_new_ball_pose()
        self.update_vis()
        self.goal_query()

    # =========================================================
    # Ball dynamics
    # =========================================================
    def calc_ball_params(self, dist, grad):
        delta = 0.0
        if self.ball_vel < 0.05:
            self.ball_vel = 0.0

        if self.ball_vel > 0.0:
            self.ball_vel *= self.ball_vel_damping

        delta = 0.0
        if self.old_dist:                     # buffer not empty
            delta = abs(self.old_dist[-1] - dist)  # previous – current
        if(dist < 100):
            self.old_dist.append(dist)        # keep newest value
            self.old_dist.pop(0)
        if len(self.old_dist) > 11:
            self.old_dist.pop(0)

        grad_norm = np.linalg.norm(grad)
        if dist < self.shot_threshold and delta > 0.1:
            self.ball_vel = min(self.ball_vel + 200.0 * delta, self.ball_vel_max)
            self.ball_direction = grad / grad_norm
            self.ball_direction[2] = 0.0

        print("Vell:", self.ball_vel, "delta:", delta, "dist", dist)
          
            # dir_norm = np.linalg.norm(self.ball_direction)
            # if dir_norm > 1e-6:
            #     self.ball_direction = self.ball_direction / dir_norm
            # else:
            #     self.ball_direction[:] = 0.0

        #self.get_logger().info(
            #f"Vel={self.ball_vel:.3f}, delta={delta:.3f}, dist={dist:.3f}, "
            #f"grad=[{grad[0]:.3f}, {grad[1]:.3f}, {grad[2]:.3f}]"
        #)

    def calc_new_ball_pose(self):
        self.ball_pose = self.ball_pose + self.update_rate * self.ball_vel * self.ball_direction
        self.ball_pose[2] = self.originz

    def update_vis(self):
        self.ball_pub.publish(
            self.create_ball(
                self.ball_pose[0],
                self.ball_pose[1],
                self.ball_pose[2],
                99
            )
        )

    # =========================================================
    # Goal / out logic
    # =========================================================
    def goal_query(self):
        # GOAL
        if (
            self.ball_pose[0] > (self.field_size[0] - 0.2)
            and -self.goal_pole_pos < self.ball_pose[1] < self.goal_pole_pos
        ):
            self.get_logger().info("GOAL !!!!!")

            self.goal_effect_pub.publish(
                self.create_goal_effect(
                    self.ball_pose[0],
                    self.ball_pose[1],
                    self.ball_pose[2],
                    777,
                    1.5
                )
            )

            self.goal_text_pub.publish(
                self.create_goal_text(
                    self.textx,
                    self.texty,
                    self.textz,
                    778,
                    "GOAL !!!"
                )
            )

            self.goal_fireworks_pub.publish(
                self.create_goal_fireworks(
                    self.ball_pose[0],
                    self.ball_pose[1],
                    self.ball_pose[2]
                )
            )

            self.reset_ball()
            self.block_until_time = self.get_clock().now() + Duration(seconds=0.5)
            return

        # OUT
        if (
            self.ball_pose[0] > self.field_size[0]
            or self.ball_pose[0] < -0.5
            or self.ball_pose[1] > self.field_size[1]
            or self.ball_pose[1] < -self.field_size[1]
        ):
            self.get_logger().info("OUT !!!!!")

            self.goal_text_pub.publish(
                self.create_goal_text(
                    self.textx,
                    self.texty,
                    self.textz,
                    778,
                    "OUT (*_*) !!!"
                )
            )

            self.reset_ball()
            self.block_until_time = self.get_clock().now() + Duration(seconds=0.5)
            return

        # waiting for user interaction
        if self.ball_vel == 0.0:
            self.goal_text_pub.publish(
                self.create_goal_text(
                    self.textx,
                    self.texty,
                    self.textz,
                    778,
                    "PUSH IT !!!"
                )
            )
            self.reset_ball()

    # =========================================================
    # Timer
    # =========================================================
    def timer_callback(self):
        if self.block_until_time is not None:
            if self.get_clock().now() < self.block_until_time:
                self.setup_goal()
                return
            self.block_until_time = None

        self.setup_goal()
        self.call_query_service()


def main(args=None):
    rclpy.init(args=args)
    node = SoccerDemoNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()