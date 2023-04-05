import argparse
import os
import sys

import torch
from scipy.io import loadmat
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from utils import (
    get_mesh,
    save_loss,
    save_res_imgs,
    save_res_objs_explict,
    save_all_imgs,
)

from AT3D.attacks.attacks import attack_implementation
from AT3D.dataset import LOADER_DICT
from AT3D.modules import Baseline
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
    model = Baseline(
        opt,
        face_veri_model=face_veri_model,
        bfm_coeff_init=bfm_coeff_init,
        mask=args.mask,
    )
    loader = LOADER_DICT["patch" if args.mask != "whole" else "whole"](
        framework="mesh",
        batch_size=args.batch_size,
        modelname=args.model,
        image_shape=img_shape,
        path_pairs=args.pairs,
        dir=args.save_dir,
    )

    loss_list = []
    if True:
        writer = None
        # with SummaryWriter(log_dir='/home/chang.liu/shape_attack/shape_attack/log/mesh') as writer:
        hyparam_dict = {
            "model": args.model,
            "eps": args.eps,
            "iters": args.iters,
            "attack": args.attack,
            "mode": args.mode,
            "method": args.method,
            "mask": args.mask,
        }
        # writer.add_hparams(run_name=hyparam_str, hparam_dict=hyparam_dict) #metric_dict={'origin cosine': loss.item(), 'adv cosine': adv_loss.item()})
        if args.mask == "whole":
            for (
                ys_feat,
                xs_align_mats,
                ids,
                exps,
                texs,
                angles,
                trans,
                gammas,
                paths,
            ) in tqdm(loader, total=len(loader)):
                face_v, face_c, tri = get_mesh(
                    mask=args.mask,
                    init=bfm_coeff_init,
                    id=ids,
                    exp=exps,
                    tex=texs,
                    angle=angles,
                    trans=trans,
                    gamma=gammas,
                )
                face_v = face_v.to(args.device)
                face_v.requires_grad = True
                face_c = face_c.to(args.device)
                face_c.requires_grad = True
                tri = tri.to(args.device)
                tri.requires_grad = False

                new_kwargs, res_imgs = attack_implementation(
                    "mesh",
                    hyparam_str,
                    None,
                    model,
                    args.attack,
                    args.iters,
                    args.mode,
                    args.eps,
                    xs_key=["vertex", "color"],
                    ys_feat=ys_feat,
                    vertex=face_v,
                    color=face_c,
                    tri=tri,
                    M=xs_align_mats,
                    align_size=(112, 112),
                    img_tensor=None,
                    writer=writer,
                )
                save_res_imgs(paths, res_imgs, hyparam_str)
                if args.save_mesh:
                    save_res_objs_explict(
                        paths, model, hyparam_str, mask=args.mask, **new_kwargs
                    )
                if args.visualize:
                    save_all_imgs(paths, xs, ys, res_imgs, hyparam_str)
        else:
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
                face_v, face_c, tri = get_mesh(
                    mask=args.mask,
                    init=bfm_coeff_init,
                    id=ids,
                    exp=exps,
                    tex=texs,
                    angle=angles,
                    trans=trans,
                    gamma=gammas,
                )
                face_v = face_v.to(args.device)
                face_v.requires_grad = True
                face_c = face_c.to(args.device)
                face_c.requires_grad = True
                tri = tri.to(args.device)
                tri.requires_grad = False

                new_kwargs, res_imgs = attack_implementation(
                    "mesh",
                    hyparam_str,
                    None,
                    model,
                    args.attack,
                    args.iters,
                    args.mode,
                    args.eps,
                    xs_key=["vertex", "color"],
                    ys_feat=ys_feat,
                    vertex=face_v,
                    color=face_c,
                    tri=tri,
                    M=xs_align_mats,
                    align_size=(112, 112),
                    img_tensor=xs,
                    writer=writer,
                    loss_list=loss_list,
                )
                save_res_imgs(paths, res_imgs, hyparam_str)
                # print(loss_list)
                if len(paths) == 1:
                    attacker, victim = paths[0].split(os.path.sep)[-2:]
                    print(attacker, victim)
                    print(
                        os.path.join(
                            args.save_dir,
                            f"{attacker}_{victim}_{args.method}_{args.model}_eps_{args.eps}_iters_{args.iters}.npy",
                        )
                    )
                    # save_loss(os.path.join(args.save_dir, f'{attacker}_{victim}_{args.method}_{args.model}_eps_{args.eps}_iters_{args.iters}.npy'), loss_list)

                if args.save_mesh:
                    save_res_objs_explict(
                        paths, model, hyparam_str, mask=args.mask, **new_kwargs
                    )
                if args.visualize:
                    save_all_imgs(paths, xs, ys, res_imgs, hyparam_str)


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
        choices=["adam", "fixed", "smooth", "others"],
    )
    parser.add_argument(
        "--method",
        help="use which attack framework",
        default="mesh",
        choices=["3dmm", "mesh"],
    )
    parser.add_argument("--pairs", help="the path of pair files")
    parser.add_argument(
        "--save_dir",
        help="the store path of the result images and meshes",
        type=str,
        default="/data/chang.liu/research/3d_adv/mesh/patch",
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
        "--save_mesh", help="save mesh or not", default=False, type=bool
    )
    parser.add_argument(
        "--visualize",
        help="save the attacker's and victim's picture together with the result",
        default=False,
        type=bool,
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    main()
