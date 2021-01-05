cmake_minimum_required(VERSION 2.8.3)
project(tele_operate)

## Add support for C++11, supported in ROS Kinetic and newer
# add_definitions(-std=c++11)
find_package(Boost COMPONENTS program_options REQUIRED)
find_package(OpenCV 4.0 COMPONENTS core highgui imgproc video REQUIRED)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
)


add_definitions(-std=c++11)# -m64) #-Wall)

## Declare a catkin package
catkin_package(
  #INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp std_msgs
  cv_bridge
  image_transport
  #DEPENDS system_lib
)


include_directories(
	${catkin_INCLUDE_DIRS}
	${Boost_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
)

link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

#add_dependencies(mavros_control_node mavros_control_messages_cpp)
#add_dependencies(mavros_comm_node mavros_control_messages_cpp)
