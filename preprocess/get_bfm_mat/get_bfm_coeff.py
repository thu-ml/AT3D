"""This script is the test script for Deep3DFaceRecon_pytorch
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import numpy as np
import torch
from models import create_model
from options.test_options import TestOptions
from PIL import Image
from scipy.io import loadmat, savemat
from util.load_mats import load_lm3d
from util.preprocess import align_img
from util.util import save_image, tensor2im
from util.visualizer import MyVisualizer


def get_data_path(root="examples"):
    im_path, lm_path, outpath = [], [], []
    for subdir in os.listdir(root):
        subdir = os.path.join(root, subdir)
        if os.path.isdir(subdir):
            for file in os.listdir(subdir):
                if (file.endswith(".jpg") or file.endswith(".png")) and (
                    not file.startswith("final_")
                ):
                    outpath.append(subdir)
                    im_path.append(os.path.join(subdir, file))
                    lm_path.append(
                        os.path.join(
                            subdir,
                            f'detection_{file.replace(".jpg",  ".txt").replace(".png", ".txt")}',
                        )
                    )
    print(im_path, lm_path)
    return im_path, lm_path, outpath


def read_data(im_path, lm_path, lm3d_std, to_tensor=True):
    # to RGB
    im = Image.open(im_path).convert("RGB")
    W, H = im.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    _, im, lm, _ = align_img(im, lm, lm3d_std)
    if to_tensor:
        im = (
            torch.tensor(np.array(im) / 255.0, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        lm = torch.tensor(lm).unsqueeze(0)
    return im, lm


def main(rank, opt, input_folder):
    device = torch.device(rank)
    torch.cuda.set_device(device)
    model = create_model(opt)
    model.setup(opt)
    model.device = device
    model.parallelize()
    model.eval()
    visualizer = MyVisualizer(opt)

    im_path, lm_path, outpath = get_data_path(input_folder)

    lm3d_std = load_lm3d(opt.bfm_folder)

    for i in range(len(im_path)):
        print(i, im_path[i])
        img_name = (
            im_path[i].split(os.path.sep)[-1].replace(".png", "").replace(".jpg", "")
        )
        if not os.path.isfile(lm_path[i]):
            continue
        im_tensor, lm_tensor = read_data(im_path[i], lm_path[i], lm3d_std)

        data = {"imgs": im_tensor, "lms": lm_tensor}
        model.set_input(data)  # unpack data from data loader

        model.test()  # run inference
        input_img_numpy = 255.0 * im_tensor.permute(0, 2, 3, 1).numpy()
        inputput_vis = torch.tensor(
            input_img_numpy / 255.0, dtype=torch.float32
        ).permute(0, 3, 1, 2)
        input_img_numpy = tensor2im(inputput_vis[0])
        save_image(
            input_img_numpy, os.path.join(outpath[i], "final_" + img_name + ".png")
        )

        model.save_coeff(
            os.path.join(outpath[i], img_name + ".mat")
        )  # save predicted coefficients


if __name__ == "__main__":
    dataset = "demo"
    opt = TestOptions().parse()  # get test options

    root_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            )
        ),
        "data",
        dataset,
        "origin",
    )
    opt.img_folder = root_dir
    opt.use_opengl = False
    opt.epoch = 20
    main(0, opt, opt.img_folder)
