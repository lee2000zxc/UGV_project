cmake_minimum_required(VERSION 2.8.3)
project(pcl_gazebo)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(Boost REQUIRED)
find_package(Eigen3 QUIET)
if(NOT EIGEN3_FOUND)
  find_package(Eigen REQUIRED)
  set(EIGEN3_INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS})
endif()
find_package(OpenCV 3.1 COMPONENTS core highgui imgproc REQUIRED)
find_package(PCL REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
  image_geometry
  tf
)

add_definitions(-std=c++11)# -m64) #-Wall)


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES test
  CATKIN_DEPENDS roscpp std_msgs
#  DEPENDS system_lib
   sensor_msgs
   cv_bridge
   image_transport
   image_geometry
   tf
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
	${catkin_INCLUDE_DIRS}
	${Boost_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${PCL_LIBARARY_DIRS})

#add_executable(stereo_vision_node src/stereo_vision_node.cpp)
add_executable(pcl_gazebo src/pcl_gazebo.cpp)


target_link_libraries(
		pcl_gazebo
        ${PCL_LIBRARIES}
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}  
        ${Boost_LIBRARIES}
)



