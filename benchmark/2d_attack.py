import argparse
import os
import sys

import torch
from scipy.io import loadmat
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AT3D.attacks.attacks import attack_implementation_2d
from AT3D.dataset import LOADER_DICT
from AT3D.modules import Pipeline
from AT3D.networks.get_model import getmodel
from AT3D.options.options import Options
from benchmark.utils import save_res_imgs


def make_hyparam_str():
    res = "model_{}_eps_{}_iters_{}_attack_{}_method_{}_bg_{}".format(
        args.model, args.eps, args.iters, args.attack, args.method, args.bg
    )
    return res


def main():
    hyparam_str = make_hyparam_str()
    print(hyparam_str)
    opt = Options()
    face_veri_model, img_shape = getmodel(args.model, device=args.device)

    # get bfm coeff
    bfm_coeff_init = loadmat(args.bfm_path)

    # get model
    model = Pipeline(
        opt,
        face_veri_model=face_veri_model,
        bfm_coeff_init=bfm_coeff_init,
        mask=args.mask,
    )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    loader = LOADER_DICT["patch_2d"](
        framework="2d",
        batch_size=args.batch_size,
        modelname=args.model,
        path_pairs=args.pairs,
        dir=args.save_dir,
        image_shape=img_shape,
    )

    for (
        xs,
        ys,
        ys_feat,
        xs_align_mats,
        ids,
        exps,
        texs,
        angles,
        trans,
        gammas,
        paths,
    ) in tqdm(loader, total=len(loader) / args.batch_size):
        pred_mask, pred_face = model.get_mask(
            id=ids, exp=exps, tex=texs, angle=angles, trans=trans, gamma=gammas
        )
        pred_face = torch.clamp(pred_face * 255.0, min=0, max=255)
        res_imgs = attack_implementation_2d(
            face_veri_model,
            "MIM",
            args.iters,
            args.eps,
            pred_mask,
            xs,
            pred_face,
            ys_feat,
            xs_align_mats,
            img_shape,
            args.eot,
        )
        save_res_imgs(paths, res_imgs, hyparam_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bfm_path",
        help="path of bfm files",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "BFM_model_front_patch.mat",
        ),
    )
    parser.add_argument("--device", help="device id", type=str, default="cuda")
    parser.add_argument(
        "--goal",
        help="dodging or impersonate",
        type=str,
        default="impersonate",
        choices=["dodging", "impersonate"],
    )
    parser.add_argument(
        "--model", help="White-box model", type=str, default="MobileFace"
    )
    parser.add_argument("--eps", help="epsilon", type=float, default=40)
    parser.add_argument("--iters", help="count of iterations", type=int, default=400)
    parser.add_argument("--seed", help="random seed", type=int, default=1234)
    parser.add_argument("--batch_size", help="batch_size", type=int, default=64)
    parser.add_argument(
        "--distance", help="l2 or linf", type=str, default="l2", choices=["linf", "l2"]
    )
    parser.add_argument("--log", help="log file", type=str, default="log.txt")
    parser.add_argument("--attack", help="attack method", type=str, default="BIM")
    parser.add_argument(
        "--mode",
        help="fixed eps or adapted eps with Adam()",
        default="adam",
        choices=["adam", "fixed"],
    )
    parser.add_argument(
        "--method",
        help="use which attack framework",
        default="2d",
        choices=["3dmm", "mesh", "2d"],
    )
    parser.add_argument("--pairs", help="the path of pair files")
    parser.add_argument(
        "--save_dir",
        help="the store path of the result images and meshes",
        type=str,
        default="",
    )
    parser.add_argument(
        "--bg", help="the factor of background deltas", type=float, default=0
    )
    parser.add_argument(
        "--mask",
        help="mask type",
        default="eye_nose",
        type=str,
        choices=[
            "whole",
            "T",
            "hole",
            "nose",
            "eye",
            "mouth",
            "cheek",
            "eye_nose",
            "respirator",
        ],
    )
    parser.add_argument("--eot", help="eot operation times", default=0, type=int)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    main()
