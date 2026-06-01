#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
import yaml
import sys

class StaticYamlPublisher(Node):
    """
    This node reads a YAML file describing static transforms
    and publishes them to the /tf_static topic.
    """
    def __init__(self, yaml_filepath):
        super().__init__('static_yaml_publisher')

        # Create the broadcaster
        self.tf_broadcaster = StaticTransformBroadcaster(self)

        # Parse the YAML and publish
        try:
            transforms = self.parse_yaml(yaml_filepath)
            if transforms:
                self.tf_broadcaster.sendTransform(transforms)
                self.get_logger().info(
                    f"Successfully published {len(transforms)} static transforms from {yaml_filepath}")
            else:
                self.get_logger().warn(f"No valid transforms found in {yaml_filepath}")
        except Exception as e:
            self.get_logger().error(f"Failed to parse or publish transforms: {e}")

    def parse_yaml(self, yaml_filepath):
        """
        Reads the YAML file and converts each entry into a TransformStamped message.
        """
        self.get_logger().info(f"Reading static transforms from {yaml_filepath}...")
        transforms_list = []
        
        # Get the current time to stamp the transforms
        now = self.get_clock().now().to_msg()

        with open(yaml_filepath, 'r') as file:
            config = yaml.safe_load(file)

            if config is None:
                self.get_logger().error(f"YAML file {yaml_filepath} is empty or invalid.")
                return []

            # Iterate over each transform defined in the YAML
            for child_frame, details in config.items():
                
                # Create a TransformStamped message
                t = TransformStamped()

                # --- Fill the header ---
                t.header.stamp = now
                t.header.frame_id = details['parent']  # The parent frame
                t.child_frame_id = child_frame         # The child frame (e.g., 'aruco_0')

                # --- Fill the translation (position) ---
                t.transform.translation.x = float(details['position']['x'])
                t.transform.translation.y = float(details['position']['y'])
                t.transform.translation.z = float(details['position']['z'])

                # --- Fill the rotation (orientation) ---
                t.transform.rotation.w = float(details['orientation']['w'])
                t.transform.rotation.x = float(details['orientation']['x'])
                t.transform.rotation.y = float(details['orientation']['y'])
                t.transform.rotation.z = float(details['orientation']['z'])

                # Add the completed transform to our list
                transforms_list.append(t)
                
                self.get_logger().info(
                    f"  -> Adding transform: {t.header.frame_id} -> {t.child_frame_id}")

        return transforms_list


def main(args=None):
    rclpy.init(args=args)

    # Check if the YAML file path was provided as an argument
    if len(sys.argv) < 2:
        print("Error: Missing YAML file path argument.")
        print("Usage: python3 static_yaml_publisher.py <path_to_your_yaml_file.yaml>")
        return

    yaml_filepath = sys.argv[1]

    try:
        # Create and spin the node
        node = StaticYamlPublisher(yaml_filepath)
        # We use spin() so the node stays alive.
        # The StaticTransformBroadcaster latches the transforms,
        # so they remain available to any new nodes (like RViz) that start later.
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        pass
    except FileNotFoundError:
        print(f"Error: The file '{yaml_filepath}' was not found.")
    finally:
        # Clean shutdown
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
