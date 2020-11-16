#!/bin/bash
t=$(date +'%Y-%m-%d-%H-%M-%S')
rosbag record -O "~/Myhal_Simulation/laserscan_test_bags/test-$t.bag" /clock /classified_points /velodyne_points /gmapping_points2 /local_planner_points2 /amcl_points2 /global_planner_points2

