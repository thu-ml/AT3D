import os

import numpy as np
import torch
from imageio import imsave

from AT3D.modules import BFM2Mesh


def save_res_imgs(paths, res_imgs, hyparam):
    for i, path in enumerate(paths):
        if os.path.exists(path) == False:
            os.makedirs(path)
        img_np = np.clip(
            res_imgs[i, :, :, :].cpu().detach().numpy().transpose(1, 2, 0), 0, 255
        ).astype(np.uint8)
        imsave(os.path.join(path, hyparam + ".png"), img_np)


def save_all_imgs(paths, xs, ys, res_imgs, hyparam):
    for i, path in enumerate(paths):
        if not os.path.exists(path):
            os.makedirs(path)
        x = np.clip(
            xs[i, :, :, :].cpu().detach().numpy().transpose(1, 2, 0), 0, 255
        ).astype(np.uint8)
        y = np.clip(
            ys[i, :, :, :].cpu().detach().numpy().transpose(1, 2, 0), 0, 255
        ).astype(np.uint8)
        img = np.clip(
            res_imgs[i, :, :, :].cpu().detach().numpy().transpose(1, 2, 0), 0, 255
        ).astype(np.uint8)
        imsave(
            os.path.join(path, "vis_" + hyparam + ".png"),
            np.concatenate((x, y, img), axis=1),
        )


def save_res_objs(paths, model, hyparam, mask, **kwargs):
    ids = kwargs["id"].detach()
    exps = kwargs["exp"].detach()
    texs = kwargs["tex"].detach()
    gammas = kwargs["gamma"].detach()
    angles = kwargs["angle"].detach()
    trans = kwargs["trans"].detach()

    for i, path in enumerate(paths):
        if os.path.exists(path) == False:
            os.makedirs(path)
        model.bfm2mesh.save_mesh(
            os.path.join(path, hyparam + ".obj"),
            mask,
            id=torch.unsqueeze(ids[i], 0),
            exp=torch.unsqueeze(exps[i], 0),
            tex=torch.unsqueeze(texs[i], 0),
            gamma=torch.unsqueeze(gammas[i], 0),
            trans=torch.unsqueeze(trans[i], 0),
            angle=torch.unsqueeze(angles[i], 0),
        )


def save_res_objs_explict(paths, model, hyparam, mask, **kwargs):
    face_v = kwargs["vertex"].detach()
    face_c = kwargs["color"].detach()
    for i, path in enumerate(paths):
        if os.path.exists(path) == False:
            os.makedirs(path)
        model.bfm2mesh.save_mesh_explict(
            os.path.join(path, hyparam + ".obj"),
            torch.unsqueeze(face_v[i], 0),
            torch.unsqueeze(face_c[i], 0),
            mask,
        )


def save_loss(path, loss_list):
    np.save(path, loss_list)


def get_mesh(mask, **kwargs):
    converter = BFM2Mesh(bfm_coeff_init=kwargs["init"])
    face_vertex, _, face_colors = converter.forward(
        kwargs["id"],
        kwargs["exp"],
        kwargs["angle"],
        kwargs["trans"],
        kwargs["tex"],
        kwargs["gamma"],
    )
    if mask == "eye":
        tri = converter.face_buf_eye
    elif mask == "eye_nose":
        tri = converter.face_buf_eye_nose
    elif mask == "respirator":
        tri = converter.face_buf_respirator
    else:
        tri = converter.face_buf
    return face_vertex.float(), face_colors, tri


def save_origin_mesh(paths, mask, **kwargs):
    converter = BFM2Mesh(bfm_coeff_init=kwargs["init"])
    for i, path in enumerate(paths):
        converter.save_mesh(
            os.path.join(path, "origin.obj"),
            mask,
            export=True,
            id=torch.unsqueeze(kwargs["id"][i], 0),
            exp=torch.unsqueeze(kwargs["exp"][i], 0),
            trans=torch.unsqueeze(kwargs["trans"][i], 0),
            angle=torch.unsqueeze(kwargs["angle"][i], 0),
            tex=torch.unsqueeze(kwargs["tex"][i], 0),
            gamma=torch.unsqueeze(kwargs["gamma"][i], 0),
        )
