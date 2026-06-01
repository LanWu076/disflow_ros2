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

 #include <IDMP/IDMP_main.h>

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <nav_msgs/msg/path.hpp>

#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.hpp>

#include <image_transport/image_transport.hpp>
#include <image_geometry/pinhole_camera_model.h>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>

#include <fstream>
#include <chrono>
#include <numeric>

#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/common/transforms.h>

namespace IDMP_ros
{

IDMPNode::IDMPNode(const rclcpp::NodeOptions& options)
{
    node_ = rclcpp::Node::make_shared("idmp", options);

    m_worldFrameId = "base_intial";

    tf2_buffer_ = std::make_unique<tf2_ros::Buffer>(node_->get_clock());
    tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);

    IDMP_ros::IDMPParam idmpParams;
    std::vector<std::string> camData;

    node_->declare_parameter<double>("idmp_rleng", idmpParams.rleng);
    node_->declare_parameter<double>("idmp_tree_hl_min", idmpParams.tree_min_hl);
    node_->declare_parameter<double>("idmp_tree_hl_max", idmpParams.tree_max_hl);
    node_->declare_parameter<double>("idmp_tree_hl_clust", idmpParams.tree_clust_hl);
    node_->declare_parameter<double>("idmp_tree_hl_init", idmpParams.tree_init_hl);
    node_->declare_parameter<double>("idmp_map_scale", idmpParams.map_scale_param);
    node_->declare_parameter<bool>("idmp_dynamic", idmpParams.dynamic);
    node_->declare_parameter<bool>("idmp_fusion", idmpParams.fusion);
    node_->declare_parameter<double>("idmp_dyn_tresh", idmpParams.dyn_tresh);
    node_->declare_parameter<double>("idmp_fus_min", idmpParams.fus_min);
    node_->declare_parameter<double>("idmp_fus_max", idmpParams.fus_max);
    node_->declare_parameter<bool>("idmp_filt_outl", filtOutl);
    node_->declare_parameter<bool>("idmp_pub_pcl", pubPcl);
    node_->declare_parameter<std::string>("idmp_world_frame", m_worldFrameId);
    node_->declare_parameter<std::string>("idmp_pcl_topic", pclTopic);
    node_->declare_parameter<std::vector<std::string>>("idmp_caminfo_topic", camData);

    node_->get_parameter("idmp_rleng", idmpParams.rleng);
    node_->get_parameter("idmp_tree_hl_min", idmpParams.tree_min_hl);
    node_->get_parameter("idmp_tree_hl_max", idmpParams.tree_max_hl);
    node_->get_parameter("idmp_tree_hl_clust", idmpParams.tree_clust_hl);
    node_->get_parameter("idmp_tree_hl_init", idmpParams.tree_init_hl);
    node_->get_parameter("idmp_map_scale", idmpParams.map_scale_param);
    node_->get_parameter("idmp_dynamic", idmpParams.dynamic);
    node_->get_parameter("idmp_fusion", idmpParams.fusion);
    node_->get_parameter("idmp_dyn_tresh", idmpParams.dyn_tresh);
    node_->get_parameter("idmp_fus_min", idmpParams.fus_min);
    node_->get_parameter("idmp_fus_max", idmpParams.fus_max);
    node_->get_parameter("idmp_filt_outl", filtOutl);
    node_->get_parameter("idmp_pub_pcl", pubPcl);
    node_->get_parameter("idmp_world_frame", m_worldFrameId);
    node_->get_parameter("idmp_pcl_topic", pclTopic);
    node_->get_parameter("idmp_caminfo_topic", camData);
    RCLCPP_INFO_STREAM(node_->get_logger(),
    "[DEBUG] camData.size() = " << camData.size());

    numCams = static_cast<int>(camData.size() / 2);

    idmp.setParams(idmpParams, numCams);

    RCLCPP_INFO_STREAM(node_->get_logger(), "Starting IDMP with:");
    RCLCPP_INFO_STREAM(node_->get_logger(), "Dynamic:\t" << idmpParams.dynamic);
    RCLCPP_INFO_STREAM(node_->get_logger(), "Fusion:\t\t" << idmpParams.fusion);
    RCLCPP_INFO_STREAM(node_->get_logger(), "Oneshot:\t\t" << idmpParams.oneshot);

    RCLCPP_INFO_STREAM(node_->get_logger(), "IDMP input:\t" << pclTopic);
    RCLCPP_INFO_STREAM(node_->get_logger(), "Number of Cameras:\t" << numCams);

    pclSub = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
        pclTopic,
        rclcpp::SensorDataQoS(),
        std::bind(&IDMPNode::pclCB, this, std::placeholders::_1));

    for (int i = 0; i < numCams; i++) {
        const std::string camInfoTopic = camData[2 * i];
        RCLCPP_INFO_STREAM(node_->get_logger(),
            "[DEBUG] subscribing camera info topic = " << camInfoTopic);
        const std::string camTfFrame = camData[2 * i + 1];

        RCLCPP_INFO_STREAM(
            node_->get_logger(),
            "IDMP Camera " << i << ":\t" << camInfoTopic
                           << "\tregistered to transform:\t" << camTfFrame);

        camInfoSubs.push_back(
            node_->create_subscription<sensor_msgs::msg::CameraInfo>(
                camInfoTopic,
                rclcpp::SensorDataQoS(),
                [this, i](sensor_msgs::msg::CameraInfo::SharedPtr msg) {
                    this->camInfoCB(msg, i);
                }));

        camTransforms.push_back(camTfFrame);
    }

    m_pclPub = node_->create_publisher<sensor_msgs::msg::PointCloud2>(
        "gp_pcl", rclcpp::QoS(1).transient_local()); // the accumulated point cloud in world frame

    m_query_svc = node_->create_service<idmp_interfaces::srv::GetDistanceGradient>(
        "query_dist_field",
        std::bind(
            &IDMPNode::queryMap,
            this,
            std::placeholders::_1,
            std::placeholders::_2)); // previous topic in IDMP

    m_distanceSlice = node_->create_publisher<sensor_msgs::msg::PointCloud2>(
        "distances", rclcpp::QoS(1)); // previous topic in IDMP

    distancePub = node_->create_publisher<sensor_msgs::msg::PointCloud2>(
        "idmp/distance_flow", rclcpp::QoS(1)); // disflow topic

    gradientPub = node_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "idmp/gradient_flow", rclcpp::QoS(1)); // gradientflow topic

    transformedCloudPub = node_->create_publisher<sensor_msgs::msg::PointCloud2>(
        "transformed_cloud", rclcpp::QoS(1));

    globalPosePub = node_->create_publisher<geometry_msgs::msg::PoseStamped>(
        "T_wc_new_pose", rclcpp::QoS(1));

    pathPub = node_->create_publisher<nav_msgs::msg::Path>(
        "trajectory", rclcpp::QoS(1).transient_local());
}

void IDMPNode::camInfoCB(
    const sensor_msgs::msg::CameraInfo::SharedPtr cam_info,
    const int camId)
{
    RCLCPP_INFO_STREAM(node_->get_logger(),
    "[camInfoCB] received camera info, camId = " << camId
    << ", width = " << cam_info->width
    << ", height = " << cam_info->height);

    m_model.fromCameraInfo(*cam_info);

    IDMP_ros::camParam c(m_model, cam_info->width, cam_info->height);
    idmp.setCam(c, camId);

    //if (camId >= 0 && camId < static_cast<int>(camInfoSubs.size())) {
        //camInfoSubs[camId].reset();
    //}
}

void IDMPNode::queryMap(
    const std::shared_ptr<idmp_interfaces::srv::GetDistanceGradient::Request> req,
    std::shared_ptr<idmp_interfaces::srv::GetDistanceGradient::Response> res)
{
    std::vector<float> queryPoints(req->points.begin(), req->points.end());

    int N_pts = queryPoints.size() / 3;

    std::vector<double> resVec;
    resVec.resize(N_pts * 8, 0);

    auto start = std::chrono::high_resolution_clock::now();

    mtx.lock();
    idmp.test(queryPoints.data(), 3, N_pts, resVec.data());
    mtx.unlock();

    std::cout << "Query: "
              << (std::chrono::high_resolution_clock::now() - start).count() * 1E-6
              << std::endl
              << std::flush;

    res->stamp = node_->now();
    res->distances.resize(N_pts, 0);
    res->gradients.resize(N_pts * 3, 0);
    res->in_bounds.resize(N_pts, 1);

    for (int index = 0; index < N_pts; index++) {
        int k8 = index * 8;

        res->distances[index] = static_cast<double>(resVec[k8]);
        res->gradients[(index * 3)] = static_cast<double>(resVec[k8 + 1]);
        res->gradients[(index * 3) + 1] = static_cast<double>(resVec[k8 + 2]);
        res->gradients[(index * 3) + 2] = static_cast<double>(resVec[k8 + 3]);
    }

    // uncomment to publish queried distance field
    // m_distanceSlice->publish(ptsToPcl(queryPoints, &resVec, m_worldFrameId));
}

Eigen::Matrix4f IDMPNode::lookupTf(
    const std::string& target_frame,
    const std::string& source_frame,
    const rclcpp::Time& time,
    const std::chrono::milliseconds timeout)
{
    try {
        geometry_msgs::msg::TransformStamped transfMsg =
            tf2_buffer_->lookupTransform(target_frame, source_frame, time, timeout);

        Eigen::Isometry3d T = tf2::transformToEigen(transfMsg);
        return T.matrix().cast<float>();
    } catch (const tf2::TransformException& e) {
        RCLCPP_ERROR_STREAM(node_->get_logger(), e.what());
        return Eigen::Matrix4f::Zero();
    }
}

void IDMPNode::pclCB(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    static int cnt = 1;
    static std::vector<double> times;
    static pcl::PointCloud<pcl::PointXYZRGB>::Ptr transfCld(new pcl::PointCloud<pcl::PointXYZRGB>);
    static pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtCld(new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(*msg, *inputCloud);

    if (msg->header.frame_id != m_worldFrameId) {
        Eigen::Matrix4f cldFrame =
            lookupTf(
                m_worldFrameId,
                msg->header.frame_id,
                rclcpp::Time(msg->header.stamp),
                std::chrono::milliseconds(100));

        if (cldFrame.isZero(0)) {
            return;
        }

        pcl::transformPointCloud(*inputCloud, *transfCld, cldFrame);
    } else {
        *transfCld = *inputCloud;
    }

    if (filtOutl) {
        pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
        sor.setInputCloud(transfCld);
        sor.setMeanK(20);
        sor.setStddevMulThresh(1.0);
        sor.filter(*filtCld);
    }

    std::vector<Eigen::Matrix4f> camPoses;

    for (auto tfFrame : camTransforms) {
        camPoses.push_back(
            lookupTf(
                m_worldFrameId,
                tfFrame,
                rclcpp::Time(msg->header.stamp),
                std::chrono::milliseconds(100)));
    }

    mtx.lock();

    auto start = std::chrono::high_resolution_clock::now();

    pcl::PointCloud<pcl::PointXYZRGB> filteredCloud;

    RCLCPP_INFO_STREAM(
        node_->get_logger(),
        "Number of testing points (raw): " << filtCld->points.size());

    Eigen::Vector3f camPos = camPoses[0].block<3, 1>(0, 3);

    filteredCloud.header.frame_id = m_worldFrameId;

    for (const auto& pt : filtCld->points) {
        float dx = pt.x - camPos.x();
        float dy = pt.y - camPos.y();
        float dz = pt.z - camPos.z();

        float distance = std::sqrt(dx * dx + dy * dy + dz * dz);

        if (distance <= 1.5f) {
            filteredCloud.points.push_back(pt);
        }
    }

    pcl::VoxelGrid<pcl::PointXYZRGB> vg;
    vg.setInputCloud(filteredCloud.makeShared());

    vg.setLeafSize(0.03f, 0.03f, 0.03f); // TODO: bring this downsample params out in YAML

    pcl::PointCloud<pcl::PointXYZRGB> downsampledCloud;
    vg.filter(downsampledCloud);

    filteredCloud = downsampledCloud;

    filteredCloud.width = static_cast<uint32_t>(filteredCloud.points.size());
    filteredCloud.height = 1;
    filteredCloud.is_dense = false;

    RCLCPP_INFO_STREAM(
        node_->get_logger(),
        "Number of testing points (filtered): " << filteredCloud.points.size());

    if (camPoses.empty()) {
        RCLCPP_ERROR_STREAM(
            node_->get_logger(),
            "camTransforms is empty; cannot get camera pose.");
        mtx.unlock();
        return;
    }

    pcl::PointCloud<pcl::PointXYZI> distCloud;

    Eigen::Matrix4f regTransform =
        (cnt > 1) ? last_T_oc : Eigen::Matrix4f::Identity();

    bool registration_ok = true;

    pcl::PointCloud<pcl::PointXYZRGB> transformedCloud;

    if (cnt > 1) {
        for (int iter = 0; iter < 20; ++iter) {
            Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();
            Eigen::Matrix<float, 6, 1> b = Eigen::Matrix<float, 6, 1>::Zero();

            pcl::transformPointCloud(filteredCloud, transformedCloud, regTransform);

            std::vector<float> queryPts;

            for (const auto& pt : transformedCloud.points) {
                queryPts.push_back(pt.x);
                queryPts.push_back(pt.y);
                queryPts.push_back(pt.z);
            }

            std::vector<double> queryResF((queryPts.size() / 3) * 8, 0.0);

            idmp.test(queryPts.data(), 3, queryPts.size() / 3, queryResF.data());

            if (iter == 0) {
                distCloud.header.frame_id = m_worldFrameId;

                for (size_t i = 0; i < transformedCloud.points.size(); ++i) {
                    int i8 = i * 8;
                    double dist = queryResF[i8];

                    pcl::PointXYZI p;
                    p.x = transformedCloud.points[i].x;
                    p.y = transformedCloud.points[i].y;
                    p.z = transformedCloud.points[i].z;
                    p.intensity = abs(dist);

                    distCloud.points.push_back(p);
                }

                sensor_msgs::msg::PointCloud2 distMsg;
                pcl::toROSMsg(distCloud, distMsg);
                distMsg.header.frame_id = m_worldFrameId;
                distMsg.header.stamp = node_->now();

                distancePub->publish(distMsg);

                visualization_msgs::msg::MarkerArray markerArray;
                int markerId = 0;

                for (size_t i = 0; i < transformedCloud.points.size(); ++i) {
                    int i8 = i * 8;

                    double gradX = queryResF[i8 + 1];
                    double gradY = queryResF[i8 + 2];
                    double gradZ = queryResF[i8 + 3];

                    geometry_msgs::msg::Point start_p;
                    geometry_msgs::msg::Point end_p;

                    start_p.x = transformedCloud.points[i].x;
                    start_p.y = transformedCloud.points[i].y;
                    start_p.z = transformedCloud.points[i].z;

                    end_p.x = start_p.x + gradX * 0.05;
                    end_p.y = start_p.y + gradY * 0.05;
                    end_p.z = start_p.z + gradZ * 0.05;

                    visualization_msgs::msg::Marker arrow;
                    arrow.header.frame_id = m_worldFrameId;
                    arrow.header.stamp = node_->now();
                    arrow.ns = "gradient_arrows";
                    arrow.id = markerId++;
                    arrow.type = visualization_msgs::msg::Marker::ARROW;
                    arrow.action = visualization_msgs::msg::Marker::ADD;
                    arrow.scale.x = 0.005;
                    arrow.scale.y = 0.01;
                    arrow.scale.z = 0.02;
                    arrow.color.r = 1.0;
                    arrow.color.g = 0.0;
                    arrow.color.b = 0.0;
                    arrow.color.a = 1.0;

                    arrow.points.push_back(start_p);
                    arrow.points.push_back(end_p);

                    markerArray.markers.push_back(arrow);
                }

                gradientPub->publish(markerArray);
            }

            for (size_t i = 0; i < transformedCloud.points.size(); ++i) {
                int i8 = i * 8;
                double dist = queryResF[i8];

                Eigen::Vector3f grad(
                    queryResF[i8 + 1],
                    queryResF[i8 + 2],
                    queryResF[i8 + 3]);

                if (grad.norm() < 1e-3) continue;
                if (std::abs(dist) > 0.15f) continue;

                Eigen::Vector3f p(
                    transformedCloud.points[i].x,
                    transformedCloud.points[i].y,
                    transformedCloud.points[i].z);

                Eigen::Matrix<float, 1, 6> J;
                J.block<1, 3>(0, 0) = grad.transpose();
                J.block<1, 3>(0, 3) = (p.cross(grad)).transpose();

                float w = std::min(1.0f, grad.norm());

                H += w * J.transpose() * J;
                b += -w * J.transpose() * dist;
            }

            if (H.determinant() < 1e-10) {
                RCLCPP_WARN_STREAM(
                    node_->get_logger(),
                    "Registration aborted: ill-conditioned system");
                registration_ok = false;
                break;
            }

            Eigen::Matrix<float, 6, 1> xi = H.ldlt().solve(b);

            float rotNorm = xi.tail<3>().norm();

            Eigen::Matrix3f dR = Eigen::Matrix3f::Identity();

            if (rotNorm > 1e-9) {
                dR = Eigen::AngleAxisf(
                         rotNorm,
                         xi.tail<3>().normalized())
                         .toRotationMatrix();
            }

            Eigen::Vector3f dt = xi.head<3>();

            Eigen::Matrix4f delta = Eigen::Matrix4f::Identity();
            delta.block<3, 3>(0, 0) = dR;
            delta.block<3, 1>(0, 3) = dt;

            regTransform = delta * regTransform;

            if (!regTransform.allFinite()) {
                RCLCPP_WARN_STREAM(
                    node_->get_logger(),
                    "regTransform became NaN/Inf; resetting and aborting.");
                regTransform = Eigen::Matrix4f::Identity();
                registration_ok = false;
                break;
            }

            if (xi.norm() < 1e-5) {
                registration_ok = true;
                break;
            }
        }
    }

    if (registration_ok) {
        sensor_msgs::msg::PointCloud2 transformed_msg;
        pcl::toROSMsg(transformedCloud, transformed_msg);
        transformed_msg.header.frame_id = m_worldFrameId;
        transformed_msg.header.stamp = node_->now();

        transformedCloudPub->publish(transformed_msg);

        Eigen::Matrix4f T_oc = regTransform;
        Eigen::Matrix4f T_co = T_oc.inverse();

        const Eigen::Matrix4f T_wc = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f T_wo = T_wc * T_co;

        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header.stamp = node_->now();
        pose_msg.header.frame_id = m_worldFrameId;

        pose_msg.pose.position.x = T_wo(0, 3);
        pose_msg.pose.position.y = T_wo(1, 3);
        pose_msg.pose.position.z = T_wo(2, 3);

        Eigen::Matrix3f R_wo = T_wo.block<3, 3>(0, 0);
        Eigen::Quaternionf q_wo(R_wo);
        q_wo.normalize();

        pose_msg.pose.orientation.x = q_wo.x();
        pose_msg.pose.orientation.y = q_wo.y();
        pose_msg.pose.orientation.z = q_wo.z();
        pose_msg.pose.orientation.w = q_wo.w();

        globalPosePub->publish(pose_msg);

        geometry_msgs::msg::PoseStamped ps;
        ps.header.stamp = node_->now();
        ps.header.frame_id = m_worldFrameId;

        ps.pose.position.x = T_wo(0, 3);
        ps.pose.position.y = T_wo(1, 3);
        ps.pose.position.z = T_wo(2, 3);

        ps.pose.orientation.x = q_wo.x();
        ps.pose.orientation.y = q_wo.y();
        ps.pose.orientation.z = q_wo.z();
        ps.pose.orientation.w = q_wo.w();

        path_msg_.header.stamp = ps.header.stamp;
        path_msg_.header.frame_id = m_worldFrameId;
        path_msg_.poses.push_back(ps);

        pathPub->publish(path_msg_);

        std::vector<Eigen::Matrix4f> camPosesNew = {T_oc};

        if(cnt > 1){
            auto testCld = idmp.processFrame(transformedCloud, camPosesNew);
            //std::cout << "[DEBUG] I am not first frame! " << std::endl;
        } else {
            pcl::transformPointCloud(filteredCloud, transformedCloud, regTransform);
            auto testCld = idmp.processFrame(transformedCloud, camPosesNew);
            //std::cout << "[DEBUG] I am first frame! " << std::endl;
        }

        last_T_oc = T_oc;

        times.push_back(
            (std::chrono::high_resolution_clock::now() - start).count() * 1E-6);

        std::cout << cnt << " "
                  << std::accumulate(times.begin(), times.end(), 0.0) / times.size()
                  << std::endl
                  << std::flush;
    }

    if (pubPcl) {
        pcl::PointCloud<pcl::PointXYZRGB> cloud;
        idmp.getAllPoints(cloud);
        cloud.header.frame_id = m_worldFrameId;

        sensor_msgs::msg::PointCloud2 cloudMsg;
        pcl::toROSMsg(cloud, cloudMsg);
        cloudMsg.header.frame_id = m_worldFrameId;
        cloudMsg.header.stamp = node_->now();

        m_pclPub->publish(cloudMsg);
    }

    mtx.unlock();
    cnt++;
}

pcl::PointCloud<pcl::PointXYZRGB>
IDMPNode::ptsToPcl(
    std::vector<float>& pts,
    std::vector<uint8_t>& col,
    std::string frame)
{
    pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
    pcl_cloud.header.frame_id = frame;
    pcl_cloud.width = pts.size() / 3;
    pcl_cloud.height = 1;
    pcl_cloud.points.resize(pts.size() / 3);

#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(pts.size() / 3); i++) {
        int i3 = i * 3;

        pcl::PointXYZRGB p;
        p.x = pts[i3];
        p.y = pts[i3 + 1];
        p.z = pts[i3 + 2];
        p.r = col[i3];
        p.g = col[i3 + 1];
        p.b = col[i3 + 2];

        pcl_cloud.points[i] = p;
    }

    return pcl_cloud;
}

sensor_msgs::msg::PointCloud2
IDMPNode::ptsToPcl(
    std::vector<float>& pts,
    std::vector<double>* queryRes,
    std::string frame)
{
    bool useDist = !(queryRes == NULL);

    sensor_msgs::msg::PointCloud2 pcl_msg;
    sensor_msgs::PointCloud2Modifier modifier(pcl_msg);

    modifier.setPointCloud2Fields(
        4,
        "x", 1, sensor_msgs::msg::PointField::FLOAT32,
        "y", 1, sensor_msgs::msg::PointField::FLOAT32,
        "z", 1, sensor_msgs::msg::PointField::FLOAT32,
        "intensity", 1, sensor_msgs::msg::PointField::FLOAT32);

    pcl_msg.header.stamp = node_->now();
    pcl_msg.header.frame_id = frame;

    pcl_msg.height = 1;
    pcl_msg.width = pts.size() / 3;
    pcl_msg.is_dense = true;

    pcl_msg.point_step = 16;
    pcl_msg.row_step = pcl_msg.point_step * pcl_msg.width;
    pcl_msg.data.resize(pcl_msg.row_step);

    sensor_msgs::PointCloud2Iterator<float> iterX(pcl_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iterY(pcl_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iterZ(pcl_msg, "z");
    sensor_msgs::PointCloud2Iterator<float> iterIntensity(pcl_msg, "intensity");

    for (int i = 0; i < static_cast<int>(pts.size() / 3); i++) {
        *iterX = pts[i * 3];
        *iterY = pts[i * 3 + 1];
        *iterZ = pts[i * 3 + 2];

        ++iterX;
        ++iterY;
        ++iterZ;

        if (useDist) {
            *iterIntensity = (*queryRes)[i * 8];
        }

        ++iterIntensity;
    }

    return pcl_msg;
}

}  // namespace IDMP_ros

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    IDMP_ros::IDMPNode node;

    rclcpp::executors::MultiThreadedExecutor spinner(
        rclcpp::ExecutorOptions(), 4);

    spinner.add_node(node.getNode());
    spinner.spin();

    rclcpp::shutdown();
    return 0;
}