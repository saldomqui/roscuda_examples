#ifndef ROS_CUDA_NODE_H
#define ROS_CUDA_NODE_H

#include <chrono>

// roscpp
#include <ros/ros.h>

#include <std_msgs/Int32.h>


// Name spaces used
using namespace std;

/**
 * The RosCuda class
 */
class RosCuda
{
public:
    /** \fn RosCuda()
     * \brief RosCuda class constructor
     */
    RosCuda();

    /** \fn ~RosCuda()
     * \brief RosCuda class destructor
     */
    ~RosCuda();

private:
    // Ros handles
    ros::NodeHandle global_nh_; // global handle used to register publications, subscriptions and services
    ros::NodeHandle local_nh_;  // local handle used to read local parameters

    ros::Subscriber dim_sub;

    void DimensionCallback(const std_msgs::Int32 msg);
};
#endif // ICARS_MAP_BUILDER_NODE_H
