# ref: https://github.com/ShawnXYang/TIP-IM/blob/master/input_diversify.py

import random

import numpy as np
import torch
import torch.nn as nn


def affine(x, vgrid, device="cuda"):
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).to(device)
    mask = nn.functional.grid_sample(mask, vgrid)
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1
    return output * mask


def Resize(x, device="cuda"):
    """
    input:
        x: (N, 299, 299)
    output:
        (N, 224, 224)
    """
    scale_factor = 2.0 / 223
    N = x.size(0)
    grid = torch.zeros((N, 224, 224, 2))
    grid[:, :, :, 0] = (
        torch.arange(0, 224, dtype=torch.float32).view((1, 1, 224)).repeat(N, 224, 1)
        * scale_factor
        - 1
    )
    grid[:, :, :, 1] = (
        torch.arange(0, 224, dtype=torch.float32).view((1, 224, 1)).repeat(N, 1, 224)
        * scale_factor
        - 1
    )
    grid = grid.to(device)
    x = x.to(device)
    return affine(x, grid, device=device)


def RandomCrop(x, device="cuda"):
    """
    input:
        x: (N, 299, 299)
    output:
        (N, 224, 224)
    """
    scale_factor = 2.0 / 223
    N = x.size(0)
    grid = torch.zeros((N, 224, 224, 2))
    start = torch.randint(0, (299 - 224) / 2, (N, 2))
    sx = start[:, 0].view(N, 1, 1).float()
    sy = start[:, 1].view(N, 1, 1).float()
    grid[:, :, :, 0] = (
        sx
        + torch.arange(0, 224, dtype=torch.float32).view((1, 1, 224)).repeat(N, 224, 1)
    ) * scale_factor - 1
    grid[:, :, :, 1] = (
        sy
        + torch.arange(0, 224, dtype=torch.float32).view((1, 224, 1)).repeat(N, 1, 224)
    ) * scale_factor - 1
    grid = grid.to(device)
    x = x.to(device)
    return affine(x, grid, device=device)


def Resize_and_padding(x, scale_factor):
    ori_size = x.size()[-2:]
    x = nn.functional.interpolate(x, scale_factor=scale_factor)
    new_size = x.size()[-2:]

    delta_w = ori_size[1] - new_size[1]
    delta_h = ori_size[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    x = nn.functional.pad(x, pad=(left, right, top, bottom), value=255)
    return x


def Rotate(x, theta, device="cuda"):
    rotation = np.zeros((2, 3, x.size(0)))
    cos = np.cos(theta).ravel()
    sin = np.sin(theta).ravel()
    rotation[0, 0] = cos
    rotation[0, 1] = sin
    rotation[1, 0] = -sin
    rotation[1, 1] = cos
    rotation = torch.Tensor(rotation.transpose((2, 0, 1))).to(device)
    grid = torch.nn.functional.affine_grid(rotation, size=x.size())
    return affine(x, grid, device)


def input_diversity_func(x, args=None):
    std_proj = random.uniform(0.01, 0.1) * 1  # if args.use_proj else 0
    std_rotate = random.uniform(0.01, 0.05) * 1  # if args.use_rotate else 0
    x = input_diversity(x, std_proj=std_proj, std_rotate=std_rotate)
    # , scale=0.,
    # cutout=False, shift=0)
    return x


def input_diversity(x, std_proj=None, std_rotate=None, device="cuda"):
    if std_proj is not None:
        n = x.size(0)
        M = np.tile(np.array([[1, 0, 0], [0, 1, 0]]), (n, 1, 1)) + np.random.normal(
            scale=std_proj, size=(n, 2, 3)
        )
        M = torch.Tensor(M)
        grid = torch.nn.functional.affine_grid(M, x.size())
        if device == "cuda":
            grid = grid.cuda()
        x = affine(x, grid, device=device)
    if std_rotate is not None:
        n = x.size(0)
        theta = np.random.normal(scale=std_rotate, size=(n, 1))
        x = Rotate(x, theta, device=device)
    return x


def matrix_diversify(x, std, device="cuda"):
    if std is not None:
        # print(x.shape)
        eps = torch.Tensor(np.random.normal(scale=std, size=x.shape))
        # print(eps)
        if device == "cuda":
            eps = eps.cuda()
        x = x + eps
        if device == "cuda":
            x = x.cuda()
    return x
