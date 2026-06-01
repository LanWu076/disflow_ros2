from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    # realsense = Node(
    #     package='realsense2_camera',
    #     executable='realsense2_camera_node',
    #     name='camera',
    #     output='screen',
    #     parameters=[{
    #         # 'color_width': 1280,
    #         # 'color_height': 720,
    #         # 'color_fps': 15,
    #         # 'pointcloud.enable': True,
    #         # 'depth_module.enable': True,
    #         # 'rgb_camera.enable': True,
    #         # 'enable_gyro': False,
    #         # 'enable_accel': False,
    #         # 'align_depth': True,
    
    #         # color
    #         'enable_color': True,
    #         'rgb_camera.color_profile': '1280x720x15',

    #         # depth
    #         'enable_depth': True,
    #         'depth_module.depth_profile': '848x480x15',

    #         # pointcloud
    #         'pointcloud.enable': True,
    #         'pointcloud.allow_no_texture_points': False,
    #         'pointcloud.stream_filter': 2,
    #         'pointcloud.stream_index_filter': 0,

    #         # align dpeth with color
    #         'align_depth.enable': True,

    #         # disable the imu due to warnings
    #         'enable_gyro': False,
    #         'enable_accel': False,

    #         # optional, reset the camera everytime
    #         # 'initial_reset': True,
    #     }]
    # )

    # static_tf = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     name='static_base_from_camera',
    #     output='screen',
    #     arguments=[
    #         '0.19240873793349833', '0.6484802776820079', '2.118238851272752',
    #         '0.8362710607211282', '0.008825442476127647',
    #         '-0.009167628415244874', '0.5481685681929417',
    #         'camera_depth_optical_frame', 'base'
    #     ]
    # )

    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_base_from_camera',
        output='screen',
        arguments=[
            '0.09637531361422165', '0.6484802776820079', '1.6008352925250686',
            '0.5464595718075703', '0.5589629162422456',
            '-0.42435546007479', '0.4570173280637886',
            'camera_depth_optical_frame', 'base'
        ]
    )

    #idmp_share = get_package_share_directory('idmp_ros')
    #params_file = os.path.join(idmp_share, 'config', 'params_realsense.yaml')

    idmp = Node(
        package='idmp_ros',
        executable='idmp',
        name='idmp',
        output='screen',
        respawn=True,
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('idmp_ros'),
                'config',
                'params_realsense.yaml'
            ])
        ]
    )

    return LaunchDescription([
        #realsense,
        static_tf,
        idmp
    ])
