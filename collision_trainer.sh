#!/bin/bash

cd src/collision_trainer
echo -e "Starting: python3 train_MyhalCollision.py $@"
python3 train_MyhalCollision.py $@

