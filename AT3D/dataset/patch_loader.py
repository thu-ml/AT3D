import os

import numpy as np
import torch

# from scipy.misc import imread
from imageio import imread

from .base_loader import BaseLoader
from .utils import read_3dmm_mat, read_align_mat, read_feat


class PatchLoader2D(BaseLoader):
    def __init__(
        self,
        framework,
        batch_size,
        modelname,
        image_shape,
        path_pairs,
        dir,
        device="cuda",
    ):
        super().__init__(batch_size, modelname, dir, device)
        self.image_shape = image_shape
        workdir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )

        with open(path_pairs, "r") as f:
            for line in f.readlines():
                sample = line.strip().split()
                assert len(sample) == 6
                sample = [os.path.join(workdir, file) for file in sample]
                self.pairs.append(sample)

    def __iter__(self):
        return super().__iter__()

    def __len__(self):
        return super().__len__()

    def __next__(self):
        if self.pos < len(self.pairs):
            minibatches_pair = self.pairs[self.pos : self.pos + self.batch_size]
            self.pos += self.batch_size
            (
                xs,
                ys,
                ys_feat,
                xs_aligned_mats,
                ys_coeff_id,
                ys_coeff_exp,
                ys_coeff_tex,
                ys_coeff_angle,
                ys_coeff_trans,
                ys_coeff_gamma,
            ) = ([], [], [], [], [], [], [], [], [], [])
            # target_paths is the target path of saving images
            target_paths = []

            for pair in minibatches_pair:
                # x means attacker, y means victim
                (
                    path_xs,
                    path_ys,
                    path_xs_align_mats,
                    path_xs_3dmm_mats,
                    path_ys_3dmm_mats,
                    path_ys_feats,
                ) = pair
                attacker = (
                    path_xs.strip()
                    .split(os.path.sep)[-1]
                    .replace(".png", "")
                    .split("_")[-1]
                )
                victim = (
                    path_ys_3dmm_mats.strip().split(os.path.sep)[-1].replace(".mat", "")
                )
                target_path = os.path.join(self.dir, attacker, victim)
                print(target_path)
                if not os.path.exists(path_xs) or not os.path.exists(path_ys):
                    continue
                x = (
                    torch.Tensor(imread(path_xs)[np.newaxis, :, :, :])
                    .permute(0, 3, 1, 2)
                    .to(self.device)
                )
                y = (
                    torch.Tensor(imread(path_ys)[np.newaxis, :, :, :])
                    .permute(0, 3, 1, 2)
                    .to(self.device)
                )
                x_M = read_align_mat(
                    path_xs_align_mats,
                    self.device,
                    keyword=f"align_{self.image_shape[0]}_{self.image_shape[1]}",
                )
                if torch.count_nonzero(x_M) == 0:
                    continue
                y_feat = read_feat(path_ys_feats, self.modelname, self.device)

                y_coeff_dict = read_3dmm_mat(
                    path_ys_3dmm_mats, self.device, need_attack=[]
                )
                x_coeff_dict = read_3dmm_mat(
                    path_xs_3dmm_mats,
                    self.device,
                    need_attack=[],
                    return_coeff=["angle", "trans"],
                )
                y_coeff_dict["angle"] = x_coeff_dict["angle"]
                y_coeff_dict["trans"] = x_coeff_dict["trans"]

                xs.append(x)
                ys.append(y)
                ys_feat.append(y_feat)
                xs_aligned_mats.append(x_M)
                ys_coeff_id.append(y_coeff_dict["id"])
                ys_coeff_exp.append(y_coeff_dict["exp"])
                ys_coeff_tex.append(y_coeff_dict["tex"])
                ys_coeff_angle.append(y_coeff_dict["angle"])
                ys_coeff_trans.append(y_coeff_dict["trans"])
                ys_coeff_gamma.append(y_coeff_dict["gamma"])
                target_paths.append(target_path)

            xs = torch.cat(xs)
            ys = torch.cat(ys)
            ys_feat = torch.cat(ys_feat)
            xs_aligned_mats = torch.cat(xs_aligned_mats)
            ys_coeff_id = torch.cat(ys_coeff_id)
            ys_coeff_exp = torch.cat(ys_coeff_exp)
            ys_coeff_tex = torch.cat(ys_coeff_tex)
            ys_coeff_angle = torch.cat(ys_coeff_angle)
            ys_coeff_trans = torch.cat(ys_coeff_trans)
            ys_coeff_gamma = torch.cat(ys_coeff_gamma)

            # xs: 224x224, ys: 112x112, ys_feat: mean ys feat of 112x112, xs' aligned matrices: Ms
            return (
                xs,
                ys,
                ys_feat,
                xs_aligned_mats,
                ys_coeff_id,
                ys_coeff_exp,
                ys_coeff_tex,
                ys_coeff_angle,
                ys_coeff_trans,
                ys_coeff_gamma,
                target_paths,
            )
        else:
            raise StopIteration


class PatchLoader(BaseLoader):
    def __init__(
        self,
        framework,
        batch_size,
        modelname,
        image_shape,
        path_pairs,
        dir,
        need_attack=[],
        device="cuda",
    ):
        super().__init__(batch_size, modelname, dir, device)
        self.image_shape = image_shape
        workdir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )

        if len(need_attack) == 0:
            if framework == "3dmm":
                self.need_attack = ["id", "exp", "tex"]
        else:
            self.need_attack = need_attack

        with open(path_pairs, "r") as f:
            for line in f.readlines():
                sample = line.strip().split()
                assert len(sample) == 6
                sample = [os.path.join(workdir, file) for file in sample]
                self.pairs.append(sample)

    def __iter__(self):
        return super().__iter__()

    def __len__(self):
        return super().__len__()

    def __next__(self):
        if self.pos < len(self.pairs):
            minibatches_pair = self.pairs[self.pos : self.pos + self.batch_size]
            self.pos += self.batch_size
            (
                xs,
                ys,
                ys_feat,
                xs_aligned_mats,
                ys_coeff_id,
                ys_coeff_exp,
                ys_coeff_tex,
                ys_coeff_angle,
                ys_coeff_trans,
                ys_coeff_gamma,
            ) = ([], [], [], [], [], [], [], [], [], [])
            # target_paths is the target path of saving images
            target_paths = []

            for pair in minibatches_pair:
                # x means attacker, y means victim
                (
                    path_xs,
                    path_ys,
                    path_xs_align_mats,
                    path_xs_3dmm_mats,
                    path_ys_3dmm_mats,
                    path_ys_feats,
                ) = pair
                attacker = (
                    path_xs.strip()
                    .split(os.path.sep)[-1]
                    .replace(".png", "")
                    .split("_")[-1]
                )
                victim = (
                    path_ys_3dmm_mats.strip().split(os.path.sep)[-1].replace(".mat", "")
                )
                target_path = os.path.join(self.dir, attacker, victim)
                print(target_path)

                x = (
                    torch.Tensor(imread(path_xs)[np.newaxis, :, :, :])
                    .permute(0, 3, 1, 2)
                    .to(self.device)
                )
                y = (
                    torch.Tensor(imread(path_ys)[np.newaxis, :, :, :])
                    .permute(0, 3, 1, 2)
                    .to(self.device)
                )
                x_M = read_align_mat(
                    path_xs_align_mats,
                    self.device,
                    keyword=f"align_{self.image_shape[0]}_{self.image_shape[1]}",
                )
                if torch.count_nonzero(x_M) == 0:
                    continue
                y_feat = read_feat(path_ys_feats, self.modelname, self.device)

                y_coeff_dict = read_3dmm_mat(
                    path_ys_3dmm_mats, self.device, self.need_attack
                )
                x_coeff_dict = read_3dmm_mat(
                    path_xs_3dmm_mats,
                    self.device,
                    self.need_attack,
                    return_coeff=["angle", "trans"],
                )
                y_coeff_dict["angle"] = x_coeff_dict["angle"]
                y_coeff_dict["trans"] = x_coeff_dict["trans"]

                xs.append(x)
                ys.append(y)
                ys_feat.append(y_feat)
                xs_aligned_mats.append(x_M)
                ys_coeff_id.append(y_coeff_dict["id"])
                ys_coeff_exp.append(y_coeff_dict["exp"])
                ys_coeff_tex.append(y_coeff_dict["tex"])
                ys_coeff_angle.append(y_coeff_dict["angle"])
                ys_coeff_trans.append(y_coeff_dict["trans"])
                ys_coeff_gamma.append(y_coeff_dict["gamma"])
                target_paths.append(target_path)

            xs = torch.cat(xs)
            ys = torch.cat(ys)
            ys_feat = torch.cat(ys_feat)
            xs_aligned_mats = torch.cat(xs_aligned_mats)
            ys_coeff_id = torch.cat(ys_coeff_id)
            ys_coeff_exp = torch.cat(ys_coeff_exp)
            ys_coeff_tex = torch.cat(ys_coeff_tex)
            ys_coeff_angle = torch.cat(ys_coeff_angle)
            ys_coeff_trans = torch.cat(ys_coeff_trans)
            ys_coeff_gamma = torch.cat(ys_coeff_gamma)

            # xs: 224x224, ys: 112x112, ys_feat: mean ys feat of 112x112, xs' aligned matrices: Ms
            return (
                xs,
                ys,
                ys_feat,
                xs_aligned_mats,
                ys_coeff_id,
                ys_coeff_exp,
                ys_coeff_tex,
                ys_coeff_angle,
                ys_coeff_trans,
                ys_coeff_gamma,
                target_paths,
            )
        else:
            raise StopIteration
