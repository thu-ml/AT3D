#!/bin/bash

set -e
set -x

env CUDA_VISIBLE_DEVICES=0

python ./benchmark/mesh_attack.py --pairs=./data/demo/pairs_patch.txt --save_dir=./data/demo/results/AT3DM/eye --mask=eye --iters=300 --model=ResNet50 --eps=3 --save_mesh=True --visualize=True
python ./benchmark/mesh_attack.py --pairs=./data/demo/pairs_patch.txt --save_dir=./data/demo/results/AT3DM/eye_nose --mask=eye_nose --iters=300 --model=ResNet50 --eps=3 --save_mesh=True --visualize=True
python ./benchmark/mesh_attack.py --pairs=./data/demo/pairs_patch.txt --save_dir=./data/demo/results/AT3DM/respirator --mask=respirator --iters=300 --model=ResNet50 --eps=3 --save_mesh=True --visualize=True
