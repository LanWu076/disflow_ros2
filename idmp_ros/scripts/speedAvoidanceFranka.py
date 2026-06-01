#!/usr/bin/env python3

import rclpy
import rclpy.node
import sys
import numpy as np
import time
import tf_transformations as transformations 
import tf2_ros.buffer
import tf2_ros.transform_listener
from rclpy.duration import Duration
from rclpy.parameter import Parameter 

# Message and Service imports
from geometry_msgs.msg import Point, Twist
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from idmp_interfaces.srv import GetDistanceGradient
from sensor_msgs.msg import PointCloud2, PointField


def pose_to_numpy(msg):
    return np.dot(
        transformations.translation_matrix(np.array([msg.position.x, msg.position.y, msg.position.z])),
        transformations.quaternion_matrix(np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]))
    )

def numpy_to_point(arr):
    if arr.shape[-1] == 4:
        arr = arr[...,:-1] / arr[...,-1]
    if len(arr.shape) == 1:
        return Point(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))
    else:
        # This is the corrected lambda function
        return np.apply_along_axis(
            lambda v: Point(x=float(v[0]), y=float(v[1]), z=float(v[2])),
            axis=-1, arr=arr
        )

def create_arrow(scale, start, end, idnum, stamp, lifetime, color=None):
    m = Marker()
    m.action = Marker.ADD
    m.header.frame_id = 'base'
    m.header.stamp = stamp
    m.lifetime = lifetime
    m.id = idnum
    m.type = Marker.ARROW
    m.pose.orientation.x = 0.0
    m.pose.orientation.y = 0.0
    m.pose.orientation.z = 0.0
    m.pose.orientation.w = 1.0
    m.scale.x = scale*0.5
    m.scale.y = scale
    m.scale.z = 0.0
    if color is None:
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 1.0
    else:
        m.color.r = float(color[0])
        m.color.g = float(color[1])
        m.color.b = float(color[2])
        m.color.a = float(color[3])
    m.points = [numpy_to_point(start), numpy_to_point(end)]
    return m

class AvoidanceController(rclpy.node.Node):

    def __init__(self):
        super().__init__(
            "apfTest",
            parameter_overrides=[Parameter(name="use_sim_time", value=False)]
        )
        
        self.buf = tf2_ros.buffer.Buffer()
        self.listener = tf2_ros.transform_listener.TransformListener(self.buf, self)

        self.marker_pub = self.create_publisher(MarkerArray, "/direction", 10)
        topic_name = "/NS_1/cartesian_velocity_follower_controller/cartesian_velocity"
        self.velocity_pub = self.create_publisher(Twist, topic_name, 10)
        self.get_logger().info(f"Publishing velocity to ROS 2 topic: {topic_name}")
        
        self.query_client = self.create_client(GetDistanceGradient, "query_dist_field")
        while not self.query_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service "query_dist_field" not available, waiting...')

        self.twist_msg = Twist()
        self.twist_msg_zero = Twist()
        self.speed_scale = 0.07
        self.goal_switch_threshold = 0.03 
        self.pose1 = np.array([0.310, 0.001, 0.320])
        self.pose2 = np.array([0.541, 0.302, 0.167])
        self.current_goal = self.pose1
        self.service_in_progress = False 
        
        self.target_frame = "fr3v2_hand_tcp"
        self.source_frame = "base"
        
        self.get_logger().info(f"Starting. Initial goal: {self.current_goal}")

        self.wait_for_tf()

        self.timer_period = 1.0 / 100.0  # 100Hz
        self.timer = self.create_timer(self.timer_period, self.main_loop_callback)
        self.get_logger().info("Controller is initialized and running at 100Hz.")


    def wait_for_tf(self):
        self.get_logger().info(f"Waiting for the first transform from '{self.source_frame}' to '{self.target_frame}'...")
        while rclpy.ok():
            try:
                if self.buf.can_transform(self.target_frame, self.source_frame, tf2_ros.Time()):
                    self.get_logger().info("TF is ready. Starting main loop.")
                    return # Exit the wait
            except tf2_ros.TransformException as e:
                self.get_logger().warn(f"Still waiting for TF: {e}")
            
            rclpy.spin_once(self, timeout_sec=0.1)
        
    def main_loop_callback(self):
        if self.service_in_progress:
            return

        try:
            transform = self.buf.lookup_transform(self.source_frame, self.target_frame, tf2_ros.Time()).transform
            tcp_pos = np.array([transform.translation.x, transform.translation.y, transform.translation.z])
            
            req = GetDistanceGradient.Request()
            req.points = [tcp_pos[0], tcp_pos[1], tcp_pos[2]]

            self.service_in_progress = True # Set the lock
            future = self.query_client.call_async(req)
            # pass tcp_pos to the callback using a lambda
            future.add_done_callback(lambda fut: self.service_response_callback(fut, tcp_pos))
            
        except Exception as e:
            self.get_logger().warn(
                f"Waiting for TF (in main loop): {e}",
                throttle_duration_sec=1.0
            )

    def service_response_callback(self, future, tcp_pos):
        try:
            res = future.result()
            if res is None:
                self.get_logger().error('Service call failed (result is None)!')
                self.service_in_progress = False # Release the lock
                return

            goalVec_raw = self.current_goal - tcp_pos
            goalDist = np.linalg.norm(goalVec_raw)

            if goalDist < self.goal_switch_threshold:
                if np.array_equal(self.current_goal, self.pose1):
                    self.current_goal = self.pose2
                else:
                    self.current_goal = self.pose1
                self.get_logger().info(f"Goal reached! Switching to: {self.current_goal}")
                
                goalVec_raw = self.current_goal - tcp_pos 
                goalDist = np.linalg.norm(goalVec_raw)

            if goalDist > 1e-3: 
                goalVec = 0.2 * (goalVec_raw / goalDist)
            else:
                goalVec = np.array([0.0, 0.0, 0.0])

            dist = res.distances[0]
            grad = np.array(res.gradients) 
            repVec = np.array([0.0, 0.0, 0.0])
            distFak = 1.0 - (4.0 * dist) # change this (reduce num) for higher reactivity to obstacles in the env.

            if not (0.0 < distFak < 1.0):
                distFak = 0.0

            repVec = grad
            resVec = distFak*repVec + (1-distFak)*goalVec

            # publish Markers
            mArr = MarkerArray()
            now = self.get_clock().now().to_msg()
            lifetime = Duration(nanoseconds=200000000).to_msg()
            
            mArr.markers.append(create_arrow(0.04,tcp_pos,tcp_pos+repVec*0.5 *distFak, 0, now, lifetime, (1,0,0,1)))
            mArr.markers.append(create_arrow(0.04,tcp_pos,tcp_pos+goalVec*0.5*(1-distFak), 1, now, lifetime, (0,0,1,1)))
            mArr.markers.append(create_arrow(0.04,tcp_pos,tcp_pos+resVec*0.5, 2, now, lifetime, (0,1,0,1)))
            self.marker_pub.publish(mArr)

            # publish Twist 
            MAX_CARTESIAN_VEL = 0.05
            res_speed = np.linalg.norm(resVec)
            if res_speed > MAX_CARTESIAN_VEL:
                resVec = resVec * (MAX_CARTESIAN_VEL / res_speed)

            self.twist_msg.linear.x = resVec[0] * self.speed_scale
            self.twist_msg.linear.y = resVec[1] * self.speed_scale
            self.twist_msg.linear.z = resVec[2] * self.speed_scale
            self.twist_msg.angular.x = 0.0
            self.twist_msg.angular.y = 0.0
            self.twist_msg.angular.z = 0.0
            self.velocity_pub.publish(self.twist_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error in service callback: {e}")
        
        self.service_in_progress = False


def main(args=sys.argv):
    rclpy.init(args=args)
    
    controller = AvoidanceController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.get_logger().info("Shutting down. Sending zero velocity.")
        if hasattr(controller, 'velocity_pub'):
            controller.velocity_pub.publish(controller.twist_msg_zero)
            time.sleep(0.1) 
        controller.destroy_node()
        rclpy.shutdown()

if __name__=="__main__":
    main()
