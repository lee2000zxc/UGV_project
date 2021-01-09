#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/point_cloud2_iterator.h>
#include <image_geometry/pinhole_camera_model.h>
#include "depth_traints.h"
#include <tf/transform_broadcaster.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>

#include <iostream>
#include <stdio.h>

using namespace cv;

int height = 480;
int width  = 640;
int mission = 0;

int flag_deptimage = 0;

float   Odom_Pos_m[3], stat_Pos_m[3];
tf::Quaternion q_map, q_stat;

sensor_msgs::CameraInfoConstPtr info_msg, back_info_msg;
sensor_msgs::ImageConstPtr Depth_msg, back_Depth_msg;

image_geometry::PinholeCameraModel model_, back_model_;
ros::Publisher pub_point_cloud, pub_point_cloud_filtered;


typedef sensor_msgs::PointCloud2 PointCloud;
namespace enc = sensor_msgs::image_encodings;

template<typename T>
void convert(
    const sensor_msgs::ImageConstPtr& depth_msg,
    PointCloud::Ptr& cloud_msg,
    const image_geometry::PinholeCameraModel& model,
    double range_max = 0.0)
{
    // Use correct principal point from calibration
    float center_x = model.cx();
    float center_y = model.cy();

    // Combine unit conversion (if necessary) with scaling by focal length for computing (X,Y)
    double unit_scaling = DepthTraits<T>::toMeters( T(1) );
    float constant_x = unit_scaling / model.fx();
    float constant_y = unit_scaling / model.fy();
    float bad_point = std::numeric_limits<float>::quiet_NaN();

    sensor_msgs::PointCloud2Iterator<float> iter_x(*cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(*cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(*cloud_msg, "z");
    const T* depth_row = reinterpret_cast<const T*>(&depth_msg->data[0]);
    int row_step = depth_msg->step / sizeof(T);
    for (int v = 0; v < (int)cloud_msg->height; ++v, depth_row += row_step)
    {
        for (int u = 0; u < (int)cloud_msg->width; ++u, ++iter_x, ++iter_y, ++iter_z)
        {
            T depth = depth_row[u];

            // Missing points denoted by NaNs
            if (!DepthTraits<T>::valid(depth))
            {
                if (range_max != 0.0)
                {
                    depth = DepthTraits<T>::fromMeters(range_max);
                }
                else
                {
                    *iter_x = *iter_y = *iter_z = bad_point;
                    continue;
                }
            }

            // Fill in XYZ
            //*iter_x = (u - center_x) * depth * constant_x;
            //*iter_y = (v - center_y) * depth * constant_y;
            //*iter_z = DepthTraits<T>::toMeters(depth);
            *iter_y = -(u - center_x) * depth * constant_x;
            *iter_z = -(v - center_y) * depth * constant_y;
            *iter_x = DepthTraits<T>::toMeters(depth);
        }
    }
}


template<typename T>
void convert_back(
    const sensor_msgs::ImageConstPtr& depth_msg,
    PointCloud::Ptr& cloud_msg,
    const image_geometry::PinholeCameraModel& model,
    double range_max = 0.0)
{
    // Use correct principal point from calibration
    float center_x = model.cx();
    float center_y = model.cy();

    // Combine unit conversion (if necessary) with scaling by focal length for computing (X,Y)
    double unit_scaling = DepthTraits<T>::toMeters( T(1) );
    float constant_x = unit_scaling / model.fx();
    float constant_y = unit_scaling / model.fy();
    float bad_point = std::numeric_limits<float>::quiet_NaN();

    sensor_msgs::PointCloud2Iterator<float> iter_x(*cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(*cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(*cloud_msg, "z");
    const T* depth_row = reinterpret_cast<const T*>(&depth_msg->data[0]);
    int row_step = depth_msg->step / sizeof(T);
    for (int v = 0; v < (int)cloud_msg->height; ++v, depth_row += row_step)
    {
        for (int u = 0; u < (int)cloud_msg->width; ++u, ++iter_x, ++iter_y, ++iter_z)
        {
            T depth = depth_row[u];

            // Missing points denoted by NaNs
            if (!DepthTraits<T>::valid(depth))
            {
                if (range_max != 0.0)
                {
                    depth = DepthTraits<T>::fromMeters(range_max);
                }
                else
                {
                    *iter_x = *iter_y = *iter_z = bad_point;
                    continue;
                }
            }

            // Fill in XYZ
            *iter_y = (u - center_x) * depth * constant_x;
            *iter_z = -(v - center_y) * depth * constant_y;
            *iter_x = -DepthTraits<T>::toMeters(depth);
        }
    }
}

void callback_image_info(const sensor_msgs::CameraInfoConstPtr& msg)
{
    info_msg = msg;
}

void callback_image_depth(const sensor_msgs::ImageConstPtr& depth_msg)
{
    Depth_msg = depth_msg;

    flag_deptimage = 1;
}

void callback_back_info(const sensor_msgs::CameraInfoConstPtr& msg)
{
    back_info_msg = msg;
}

void callback_back_depth(const sensor_msgs::ImageConstPtr& depth_msg)
{
    back_Depth_msg = depth_msg;

    flag_deptimage = 1;
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "pcl_convertor");
    ros::NodeHandle nh_sub;
    ros::NodeHandle nh_pub;

    image_transport::ImageTransport it_sub(nh_sub);
    image_transport::Subscriber     image_sub_depth, back_sub_depth;

    image_sub_depth = it_sub.subscribe("/camera/depth/image_raw", 1, &callback_image_depth);
    back_sub_depth = it_sub.subscribe("/back/depth/image_raw", 1, &callback_back_depth);
    ros::Subscriber image_sub_info = nh_sub.subscribe("/camera/depth/camera_info", 1, &callback_image_info);
    ros::Subscriber back_sub_info = nh_sub.subscribe("/back/depth/camera_info", 1, &callback_back_info);

    //image_sub_depth = it_sub.subscribe("/camera/depth/image_rect_raw", 1, &callback_image_depth);
    //ros::Subscriber image_sub_info = nh_sub.subscribe("/camera/depth/camera_info", 1, &callback_image_info);

    pub_point_cloud = nh_pub.advertise<sensor_msgs::PointCloud2>("/camera/pointcloud", 1);
    //pub_point_cloud_filtered = nh_pub.advertise<sensor_msgs::PointCloud2>("/camera/pointcloud_filtered", 1);
    pub_point_cloud_filtered = nh_pub.advertise<sensor_msgs::PointCloud2>("/velodyne_points", 1);

    ros::Rate loop_rate(20);
    while (ros::ok())
    {
        ros::spinOnce();

        if (flag_deptimage == 1)
        {
            PointCloud::Ptr cloud_msg(new PointCloud);
            cloud_msg->header = Depth_msg->header;
            cloud_msg->header.frame_id = "camera_link";
            cloud_msg->height = Depth_msg->height;
            cloud_msg->width  = Depth_msg->width;
            cloud_msg->is_dense = false;
            cloud_msg->is_bigendian = false;

            sensor_msgs::PointCloud2Modifier pcd_modifier(*cloud_msg);
            pcd_modifier.setPointCloud2FieldsByString(1, "xyz");

            model_.fromCameraInfo(info_msg);
            if (Depth_msg->encoding == enc::TYPE_16UC1)
            {
                convert<uint16_t>(Depth_msg, cloud_msg, model_);
            }
            else if (Depth_msg->encoding == enc::TYPE_32FC1)
            {
                convert<float>(Depth_msg, cloud_msg, model_);
            }
            pub_point_cloud.publish(cloud_msg);

            pcl::PointCloud<pcl::PointXYZ> cloud, cloud_back;
            pcl::PointCloud<pcl::PointXYZ> cloud_filtered, cloud_filtered_back;

            sensor_msgs::PointCloud2 output;

            pcl::fromROSMsg(*cloud_msg, cloud);

            //pcl::StatisticalOutlierRemoval<pcl::PointXYZ> statFilter;
            //statFilter.setInputCloud(cloud.makeShared());
            //statFilter.setMeanK(20);
            //statFilter.setStddevMulThresh(0.2);
            //statFilter.filter(cloud_filtered);

            pcl::VoxelGrid<pcl::PointXYZ> voxelSampler;
            voxelSampler.setInputCloud(cloud.makeShared());
            voxelSampler.setLeafSize(0.05f,0.05f,0.05f );
            //voxelSampler.setLeafSize(0.1f,0.1f,0.1f );
            //voxelSampler.setLeafSize(0.2f,0.2f,0.2f );
            voxelSampler.filter(cloud_filtered);

            pcl::PassThrough<pcl::PointXYZ> pass;
            pass.setInputCloud(cloud_filtered.makeShared());
            pass.setFilterFieldName ("x");
            pass.setFilterLimits (0.0, 10.0);
            //pass.setFilterLimitsNegative (true);
            pass.filter(cloud_filtered);




            // back cloud
            PointCloud::Ptr back_cloud_msg(new PointCloud);
            back_cloud_msg->header = back_Depth_msg->header;
            back_cloud_msg->header.frame_id = "back_link";
            back_cloud_msg->height = back_Depth_msg->height;
            back_cloud_msg->width  = back_Depth_msg->width;
            back_cloud_msg->is_dense = false;
            back_cloud_msg->is_bigendian = false;

            sensor_msgs::PointCloud2Modifier pcd_modifier_back(*back_cloud_msg);
            pcd_modifier_back.setPointCloud2FieldsByString(1, "xyz");

            back_model_.fromCameraInfo(back_info_msg);
            if (back_Depth_msg->encoding == enc::TYPE_16UC1)
            {
                convert_back<uint16_t>(back_Depth_msg, back_cloud_msg, back_model_);
            }
            else if (back_Depth_msg->encoding == enc::TYPE_32FC1)
            {
                convert_back<float>(back_Depth_msg, back_cloud_msg, back_model_);
            }
            //pub_point_cloud_back.publish(back_cloud_msg);

            pcl::fromROSMsg(*back_cloud_msg, cloud_back);

            //pcl::VoxelGrid<pcl::PointXYZ> voxelSampler;
            voxelSampler.setInputCloud(cloud_back.makeShared());
            voxelSampler.setLeafSize(0.05f,0.05f,0.05f );
            //voxelSampler.setLeafSize(0.1f,0.1f,0.1f );
            //voxelSampler.setLeafSize(0.2f,0.2f,0.2f );
            voxelSampler.filter(cloud_filtered_back);


            cloud_filtered += cloud_filtered_back;

            pcl::toROSMsg(cloud_filtered, output);
            pub_point_cloud_filtered.publish(output);
        }

        loop_rate.sleep();
    }

    return 0;
}
