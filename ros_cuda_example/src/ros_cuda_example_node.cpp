/************************************************************************
 * by Salvador Dominguez (from IRCCyN (ECN))
 *
 * Copyright (C) 2014
 *
 * Salvador Dominguez Quijada
 * salvador.dominguez@sagarobotics.com
 *
 * ros_cuda_example is licenced under the GPL License
 *
 * You are free:
 *   - to Share - to copy, distribute and transmit the work
 *   - to Remix - to adapt the work
 *
 * Under the following conditions:
 *
 *   - Attribution. You may mention the ICARS project and the authors
 *
 *   - Noncommercial. You may not use this work for commercial purposes.
 *
 *   - Share Alike. If you alter, transform, or build upon this work,
 *     you may distribute the resulting work only under the same or
 *     similar license to this one.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 ************************************************************************/
/**
\file    ros_cuda_example_node.cpp
\brief   Example of how to integrate ROS and CUDA
\author  Salvador Dominguez
\date    13/09/2022
*/

// roscpp
#include "ros/ros.h"

#include "RosCuda.h"

using namespace std;

RosCuda *rcu;//The pointer to the cuda processing class


//The main of the amcl node
int main(int argc, char** argv)
{
    ros::init(argc, argv, "ros_cuda_example");

    rcu=new RosCuda();

    ros::spin();

    delete(rcu);

    return(0);
}
