#!/bin/bash

set -e
set -x

python ./preprocess/get_bfm_mat/get5landmarks.py
cp ./preprocess/get_bfm_mat/get_bfm_coeff.py ./preprocess/get_bfm_mat/Deep3DFaceRecon_pytorch/get_bfm_coeff.py
cd ./preprocess/get_bfm_mat/Deep3DFaceRecon_pytorch
if [ ! -d "./insightface/" ]; then
    git clone https://github.com/deepinsight/insightface.git
    cp -r ./insightface/recognition/arcface_torch ./models/
fi

if [ ! -d "./nvdiffrast/" ]; then
    git clone https://github.com/NVlabs/nvdiffrast
fi

cd ./nvdiffrast
git checkout fad71a4
pip install .

cd ..

python ./get_bfm_coeff.py

cd ../../..
python ./preprocess/process_input.py

echo "Preprocess success!"