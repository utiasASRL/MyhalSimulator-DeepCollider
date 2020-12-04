#!/bin/bash

source ~/catkin_ws/devel/setup.bash

pkill roscore
pkill rosmaster
pkill gzclient
pkill gzserver
pkill rviz

echo "Waiting"

until rostopic list; do sleep 0.5; done #wait until rosmaster has started

echo "Go"

roslaunch classifier online_frame_preds.launch

