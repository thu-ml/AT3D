#!/bin/bash

set -e
set -x

env CUDA_VISIBLE_DEVICES=0

python ./benchmark/2d_attack.py --mask=eye --eps=40 --iters=400 --model=ResNet50 --pairs=./data/demo/pairs_patch.txt --save_dir=./data/demo/results/2D_MIM/eye --visualize=True
python ./benchmark/2d_attack.py --mask=eye_nose --eps=40 --iters=400 --model=ResNet50 --pairs=./data/demo/pairs_patch.txt --save_dir=./data/demo/results/2D_MIM/eye_nose --visualize=True
python ./benchmark/2d_attack.py --mask=respirator --eps=40 --iters=400 --model=ResNet50 --pairs=./data/demo/pairs_patch.txt --save_dir=./data/demo/results/2D_MIM/respirator --visualize=True
