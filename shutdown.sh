#!/bin/bash

rosnode kill -a
pkill roscore
pkill gzclient
pkill gzserver
pkill rosmaster
pkill rviz
