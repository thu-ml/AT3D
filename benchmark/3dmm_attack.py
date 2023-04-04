import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse

import torch
from scipy.io import loadmat
from tqdm import tqdm
from utils import save_loss, save_res_imgs, save_res_objs

from AT3D.attacks.attacks import attack_implementation
from AT3D.dataset import LOADER_DICT
from AT3D.modules import Pipeline
from AT3D.networks.get_model import getmodel
from AT3D.options.options import Options


def make_hyparam_str():
    res = "{}_{}_model_{}_eps_{}_iters_{}_attack_{}_mode_{}_bg_{}".format(
        args.method,
        args.mask,
        args.model,
        args.eps,
        args.iters,
        args.attack,
        args.mode,
        args.bg,
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

    loader = LOADER_DICT["patch" if args.mask != "whole" else "whole"](
        framework="3dmm",
        batch_size=args.batch_size,
        modelname=args.model,
        path_pairs=args.pairs,
        dir=args.save_dir,
        image_shape=img_shape,
        need_attack=args.need_attack,
    )

    loss_list = []
    if args.mask == "whole":
        # 重建的 render 需要 224x224，人脸识别模块需要 112x112
        for (
            xs,
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
            new_kwargs, res_imgs = attack_implementation(
                "3dmm",
                hyparam_str,
                None,
                model,
                args.attack,
                args.iters,
                args.mode,
                args.eps,
                xs_key=args.need_attack,
                ys_feat=ys_feat,
                id=ids,
                exp=exps,
                angle=angles,
                trans=trans,
                tex=texs,
                gamma=gammas,
                M=xs_align_mats,
                align_size=img_shape,
                img_tensor=xs,
            )
            save_res_imgs(paths, res_imgs, hyparam_str)
            if args.save_mesh:
                save_res_objs(paths, model, hyparam_str, mask=args.mask, **new_kwargs)
    else:
        for (
            xs,
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
            new_kwargs, res_imgs = attack_implementation(
                "3dmm",
                hyparam_str,
                None,
                model,
                args.attack,
                args.iters,
                args.mode,
                args.eps,
                xs_key=args.need_attack,
                ys_feat=ys_feat,
                id=ids,
                exp=exps,
                angle=angles,
                trans=trans,
                tex=texs,
                gamma=gammas,
                M=xs_align_mats,
                align_size=img_shape,
                img_tensor=xs,
                loss_list=loss_list,
            )
            save_res_imgs(paths, res_imgs, hyparam_str)
            if len(paths) == 1:
                attacker, victim = paths[0].split(os.path.sep)[-2:]
                print(
                    os.path.join(
                        args.save_dir,
                        f"{attacker}_{victim}_{args.method}_{args.model}_eps_{args.eps}_iters_{args.iters}.npy",
                    )
                )
                # save_loss(os.path.join(args.save_dir, f'{attacker}_{victim}_{args.method}_{args.model}_eps_{args.eps}_iters_{args.iters}.npy'), loss_list)

            if args.save_mesh:
                save_res_objs(paths, model, hyparam_str, mask=args.mask, **new_kwargs)


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
    parser.add_argument("--eps", help="epsilon", type=float, default=3)
    parser.add_argument("--iters", help="count of iterations", type=int, default=200)
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
        default="3dmm",
        choices=["3dmm", "mesh"],
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
    parser.add_argument(
        "--need_attack",
        help="the coeffs that need attacking",
        type=list,
        default=["id", "exp", "tex"],
    )
    parser.add_argument(
        "--save_mesh", help="save mesh or not", default=False, type=bool
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    main()
