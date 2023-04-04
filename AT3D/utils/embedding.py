import os
import sys

from imageio import imread

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../align_methods")

import numpy as np
import torch

from AT3D.networks.get_model import getmodel
from AT3D.utils.align import align_image


def infer(img_path, model_name="ArcFace"):
    """
    White box infer
    """
    img_mat = imread(img_path).astype(np.float32)
    model, img_shape = getmodel(model_name)
    img = align_image(img_mat, img_shape)[0][np.newaxis, :, :, :]
    img = torch.Tensor(img).permute(0, 3, 1, 2)

    img = img.cuda()
    model = model.cuda()
    return model.forward(img)


def save_embedding(dir, save_prefix, model_names):
    if os.path.exists(save_prefix) == False:
        os.makedirs(save_prefix)
    for file in os.listdir(dir):
        if file.endswith(".png") and file.startswith("final_"):
            path = os.path.join(dir, file)

            embeddings = []
            for model_name in model_names:
                embeddings.append(infer(path, model_name))

            mapp = dict()
            for model_name, embedding in zip(model_names, embeddings):
                mapp[model_name] = embedding.detach().cpu()

            np.savez(
                os.path.join(
                    save_prefix,
                    f"embedding_{file.replace('.png', '').replace('.jpg', '')}.npz",
                ),
                **mapp,
            )
