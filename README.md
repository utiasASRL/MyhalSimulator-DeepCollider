# Intro

Classification code is in `src/classifier/src/online_frame_preds.py`

cpp wrappers located in `src/classifier/src/cpp_wrappers` need to be compiled like in KPConv original repo


# Usage

## Double docker script

To use online classifications, we need two different containers. One running the simulation of th `docker_ros_melodic` image and oine running the Deep prediction on the `docker_ros_noetic` image. To do this we use the double_docker.sh script: 

```bash
cd /EXP_ROOT_PATH/MyhalSimulator-docker/
./double_docker.sh -c "./master.sh -OPTIONS"
```

The noetic docker will be detrached and you will see the output of the melodic docker. For example, a common call is:

+ `./double_docker.sh -c "./master.sh -fve -m 2 -t A_tour -p Sc1_params"`

## Choosing the network for predictions

You can change parameters in the launch file: `src/classifier/launch/online_frame_preds.launch`





