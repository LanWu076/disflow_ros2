/*
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License v3 as published by
 *    the Free Software Foundation.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License v3 for more details.
 *
 *    You should have received a copy of the GNU General Public License v3
 *    along with this program.  If not, see https://www.gnu.org/licenses/gpl-3.0.html.
 *
 *    Authors: Lan Wu <Lan.Wu-2@uts.edu.au>
 */

#ifndef IDMP_MAIN_H
#define IDMP_MAIN_H

#include "../IDMP/IDMP.h"

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.hpp>

#include <image_transport/image_transport.hpp>
#include <image_geometry/pinhole_camera_model.h>

#include <std_srvs/srv/empty.hpp>
#include <idmp_interfaces/srv/get_distance_gradient.hpp>

#include <nav_msgs/msg/path.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Dense>

#include <mutex>
#include <memory>
#include <vector>
#include <string>
#include <chrono>

namespace IDMP_ros
{
    class IDMPNode
    {
    public:
        explicit IDMPNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

        rclcpp::Node::SharedPtr getNode() const
        {
            return node_;
        }

    private:
        void camInfoCB(
            const sensor_msgs::msg::CameraInfo::SharedPtr info_msg,
            const int camId);

        void pclCB(
            const sensor_msgs::msg::PointCloud2::SharedPtr msg);

        void queryMap(
            const std::shared_ptr<idmp_interfaces::srv::GetDistanceGradient::Request> req,
            std::shared_ptr<idmp_interfaces::srv::GetDistanceGradient::Response> res);

        sensor_msgs::msg::PointCloud2 ptsToPcl(
            std::vector<float>& pts,
            std::vector<double>* queryRes,
            std::string frame);

        pcl::PointCloud<pcl::PointXYZRGB> ptsToPcl(
            std::vector<float>& pts,
            std::vector<uint8_t>& col,
            std::string frame);

        Eigen::Matrix4f lookupTf(
            const std::string& target_frame,
            const std::string& source_frame,
            const rclcpp::Time& time,
            const std::chrono::milliseconds timeout);

    private:
        rclcpp::Node::SharedPtr node_;

        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_pclPub;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_distanceSlice;

        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr distancePub;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr gradientPub;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr transformedCloudPub;
        rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr globalPosePub;
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pathPub;

        nav_msgs::msg::Path path_msg_;

        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pclSub;
        rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camInfoSub;
        std::vector<rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr> camInfoSubs;

        std::string m_worldFrameId; 
        std::string pclTopic;

        int numCams;
        std::vector<std::string> camTransforms;

        image_geometry::PinholeCameraModel m_model;

        rclcpp::Service<idmp_interfaces::srv::GetDistanceGradient>::SharedPtr m_query_svc;

        std::unique_ptr<tf2_ros::Buffer> tf2_buffer_;
        std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;

        IDMP_ros::IDMP idmp;

        std::vector<double> pRes;
        bool filtOutl;
        bool pubPcl;
        std::mutex mtx;

        Eigen::Matrix4f last_T_wc = Eigen::Matrix4f::Identity();  // camera -> world
        Eigen::Matrix4f last_T_oc = Eigen::Matrix4f::Identity();  // camera -> object
    };

}

#endif