import os
import sys

import cv2
from imageio import imread

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../align_methods")

import kornia
import numpy as np
import torch

from AT3D.align_methods.align import align


def get_aligned(img, img_shape):
    # get matrix M by align methods
    _, M = align(img, img_shape)
    # w, h
    img = torch.from_numpy(img[np.newaxis, :, :, :]).permute(0, 3, 1, 2)
    M = torch.from_numpy(M.astype(np.float32)).unsqueeze(0)
    return kornia.warp_affine(img, M, (img_shape[1], img_shape[0])), M


def align_image(img_mat, img_shape=(112, 112), filename="./example.png"):
    """
    Align image with specific shape
    """

    if img_shape[0] == img_shape[1]:
        img, M = align(img_mat)
    else:
        img, M = align(img_mat, img_shape)

    img_tensor = torch.Tensor(img[np.newaxis, :, :, :]).permute(0, 3, 1, 2)
    # std_proj = random.uniform(0.1, 0.2)
    # std_rotate = random.uniform(0.01, 0.02)
    # input_diversity(img_tensor, std_proj, std_rotate, 'cpu')
    img = img_tensor.permute(0, 2, 3, 1).squeeze(0).numpy().astype(np.uint8)
    if img_shape == (160, 160) or img_shape == (600, 600):
        img = cv2.resize(img, img_shape, img)
    # imsave(filename, img)

    return img, M


def save_align_matrix(dir, save_prefix, image_shapes):
    if os.path.exists(save_prefix) == False:
        os.makedirs(save_prefix)
    for file in os.listdir(dir):
        if file.endswith(".png") and file.startswith("final_"):
            path = os.path.join(dir, file)
            img_mat = imread(path).astype(np.float32)
            Ms = []
            for image_shape in image_shapes:
                fig, M = align_image(img_mat, image_shape)
                if M is None:
                    print(path, file=sys.stderr)
                Ms.append(M)

            mapp = dict()
            for image_shape, M in zip(image_shapes, Ms):
                mapp[f"align_{image_shape[0]}_{image_shape[1]}"] = M
            np.savez(
                os.path.join(
                    save_prefix,
                    f"align_{file.replace('.png', '').replace('.jpg', '')}.npz",
                ),
                **mapp,
            )
