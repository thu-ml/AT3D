import warnings

import numpy as np
import torch
import torch.optim
from torch import linalg as LA

from AT3D.attacks.base import ConstrainedMethod

from pytorch3d.structures import Meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
)


class BIM(ConstrainedMethod):
    def __init__(self, framework, model, goal, distance_metric, eps, iters=100):
        super(BIM, self).__init__(model, goal, distance_metric, eps)
        self.iters = iters
        self.framework = framework

    def join_batch_mesh(self, face_vs, tri):
        """Get a batch of mesh from vs"""
        tri = tri.unsqueeze(0).repeat_interleave(face_vs.shape[0], dim=0)
        meshes = Meshes(verts=face_vs, faces=tri)

        return meshes

    def getLaplacianLoss(self, face_vs, tri, with_edge=True):
        """Laplacian loss for geometry"""
        meshes = self.join_batch_mesh(face_vs, tri)
        return 0.5 * mesh_laplacian_smoothing(meshes=meshes, method="uniform") + (
            0.2 * mesh_edge_loss(meshes) if with_edge else 0
        )

    def getChamferLoss(self, face_vs_old, face_vs):
        return chamfer_distance(face_vs_old, face_vs)[0]

    def batch_attack(self, hyparam_str, target_name, mode, xs_key, ys_feat, **kwargs):
        kwargs_old = {key: kwargs[key].clone().detach() for key in xs_key}
        for key in xs_key:
            kwargs[key] = kwargs[key].clone().detach().requires_grad_(True)

        if "loss_list" in kwargs.keys():
            loss_list = kwargs["loss_list"]
        else:
            loss_list = []

        if self.framework == "3dmm":
            if mode == "fixed":
                for i in range(self.iters):
                    for key in xs_key:
                        features, res_imgs = self.model.forward(
                            "{}_{}_{}".format(target_name, hyparam_str, i), **kwargs
                        )
                        loss = self.getLoss(features, ys_feat)
                        loss_list.append(loss.item())
                        loss.backward(retain_graph=True)
                        grad = kwargs[key].grad
                        self.model.zero_grad()
                        kwargs[key] = self.step(
                            kwargs[key],
                            1.5 * self.eps / self.iters,
                            grad,
                            kwargs_old[key],
                            self.eps,
                        )
                        kwargs[key] = kwargs[key].detach().requires_grad_(True)
            elif mode == "adam":
                if self.iters == 0:
                    return kwargs, []
                opt = torch.optim.Adam(
                    params=[kwargs[key].requires_grad_(True) for key in xs_key],
                    lr=1.5 * self.eps / self.iters,
                )
                min_loss = 1000
                earlystop = 0
                for i in range(self.iters):
                    opt.zero_grad()
                    (
                        face_v_olds,
                        _,
                        face_c_olds,
                    ) = self.model.bfm2mesh.computer_for_render(**kwargs)
                    features, res_imgs = self.model.forward(prefix=None, **kwargs)
                    loss = self.getLoss(features, ys_feat)
                    loss_list.append(loss.item())
                    loss.backward(retain_graph=True)
                    self.model.zero_grad()
                    opt.step()
                    (
                        face_vertex,
                        _,
                        face_color,
                    ) = self.model.bfm2mesh.computer_for_render(**kwargs)
                    # print(f"v 3dmm: {face_vertex - face_v_olds}")
                    # if min_loss > loss.item():
                    #   min_loss = loss.item()
                    #   earlystop = 0
                    # else:
                    #   earlystop += 1
                    # if earlystop >= 20:
                    #   break
            else:
                print("Unknown mode! Nothing will be done!")

        else:
            face_v_olds = kwargs["vertex"].clone().detach()
            opt = torch.optim.Adam(
                params=[kwargs[key].requires_grad_(True) for key in xs_key],
                lr=0.0015 * self.eps / self.iters,
            )
            for i in range(self.iters):
                face_v_olds = kwargs["vertex"].clone().detach()
                opt.zero_grad()
                features, res_imgs = self.model.forward(prefix=None, **kwargs)
                loss = self.getLoss(features, ys_feat)
                if mode == "smooth":
                    loss = (
                        loss
                        + self.getLaplacianLoss(
                            kwargs["vertex"], kwargs["tri"], with_edge=True
                        )
                        + 1.0 * self.getChamferLoss(face_v_olds, kwargs["vertex"])
                    )
                loss_list.append(loss.item())
                loss.backward(retain_graph=True)
                self.model.zero_grad()
                opt.step()

        return kwargs, res_imgs
