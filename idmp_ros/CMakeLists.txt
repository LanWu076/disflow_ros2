cmake_minimum_required(VERSION 3.8)
project(idmp_ros)

# ===== Toolchain / Flags =====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -ggdb -fopenmp -fopenacc")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fopenmp")

message(STATUS "Using package.xml at: ${CMAKE_CURRENT_SOURCE_DIR}/package.xml")

# ===== Ament base =====
find_package(ament_cmake REQUIRED)

# ===== External interfaces (separate package) =====
find_package(idmp_interfaces REQUIRED)

# ===== ROS deps =====
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(nav_msgs REQUIRED)

find_package(depth_image_proc QUIET)
find_package(image_geometry REQUIRED)
find_package(pcl_conversions REQUIRED)

# ===== 3rd-party =====
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread system program_options)

# ===== Include dirs =====
include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

# ===== Targets =====
add_executable(idmp
  src/IDMP_main.cpp
  src/IDMP/gp.cpp
  src/IDMP/octree.cpp
  src/IDMP/IDMP.cpp
)

ament_target_dependencies(idmp
  rclcpp rclcpp_components
  sensor_msgs std_srvs visualization_msgs message_filters
  tf2 tf2_ros tf2_eigen tf2_geometry_msgs nav_msgs
  image_transport image_geometry pcl_conversions
  idmp_interfaces
)

target_link_libraries(idmp
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
)

# ===== Python tools =====
install(PROGRAMS scripts/queryToolRos2.py scripts/static_yaml_publisher.py scripts/speedAvoidanceFranka.py scripts/dragonBallDemo.py
  DESTINATION lib/${PROJECT_NAME}
)

# ===== Install =====

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS idmp
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
# ===== Export =====
ament_export_dependencies(
  idmp_interfaces            
  rosidl_default_runtime     
  rclcpp
)

ament_package()
