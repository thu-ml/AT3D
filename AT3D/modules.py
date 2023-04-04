import random

import cv2
import kornia
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from skimage.io import imread, imsave

from AT3D.utils.input_diversify import input_diversity, matrix_diversify
from AT3D.utils.nvdiffrast import MeshRenderer


def perspective_projection(focal, center):
    return (
        np.array([focal, 0, center, 0, focal, center, 0, 0, 1])
        .reshape([3, 3])
        .astype(np.float32)
        .transpose()
    )


class SH:
    def __init__(self):
        self.a = [np.pi, 2 * np.pi / np.sqrt(3.0), 2 * np.pi / np.sqrt(8.0)]
        self.c = [
            1 / np.sqrt(4 * np.pi),
            np.sqrt(3.0) / np.sqrt(4 * np.pi),
            3 * np.sqrt(5.0) / np.sqrt(12 * np.pi),
        ]


# ref: https://github.com/sicxu/Deep3DFaceRecon_pytorch/blob/master/models/bfm.py#L274
class BFM2Mesh(nn.Module):
    def __init__(
        self,
        bfm_coeff_init,
        focal=1015.0,
        center=112.0,
        camera_distance=10.0,
        init_lit=np.array([0.8, 0, 0, 0, 0, 0, 0, 0, 0]),
        recenter=True,
    ):
        super().__init__()
        # mean face shape. [3*N,1]
        self.mean_shape = bfm_coeff_init["meanshape"].astype(np.float32)
        # identity basis. [3*N,80]
        self.id_base = bfm_coeff_init["idBase"].astype(np.float32)
        # expression basis. [3*N,64]
        self.exp_base = bfm_coeff_init["exBase"].astype(np.float32)
        # mean face texture. [3*N,1] (0-255)
        self.mean_tex = bfm_coeff_init["meantex"].astype(np.float32)
        # texture basis. [3*N,80]
        self.tex_base = bfm_coeff_init["texBase"].astype(np.float32)
        # face indices for each vertex that lies in. starts from 0. [N,8]
        self.point_buf = bfm_coeff_init["point_buf"].astype(np.int64) - 1

        # vertex indices for each face. starts from 0. [F,3]
        self.face_buf = bfm_coeff_init["tri"].astype(np.int64) - 1
        self.face_buf_T = bfm_coeff_init["tri_T"].astype(np.int64) - 1
        self.face_buf_crop_eye = bfm_coeff_init["tri_crop_eye"].astype(np.int64) - 1
        self.face_buf_nose = bfm_coeff_init["tri_nose"].astype(np.int64) - 1
        self.face_buf_eye = bfm_coeff_init["tri_eye"].astype(np.int64) - 1
        self.face_buf_mouth = bfm_coeff_init["tri_mouth"].astype(np.int64) - 1
        self.face_buf_cheek = bfm_coeff_init["tri_cheek"].astype(np.int64) - 1
        self.face_buf_eye_nose = bfm_coeff_init["tri_eye_nose"].astype(np.int64) - 1
        self.face_buf_respirator = bfm_coeff_init["tri_respirator"].astype(np.int64) - 1

        # vertex indices for 68 landmarks. starts from 0. [68,1]
        self.keypoints = np.squeeze(bfm_coeff_init["keypoints"]).astype(np.int64) - 1

        if recenter:
            mean_shape = self.mean_shape.reshape([-1, 3])
            mean_shape = mean_shape - np.mean(mean_shape, axis=0, keepdims=True)
            self.mean_shape = mean_shape.reshape([-1, 1])

        self.persc_proj = perspective_projection(focal, center)
        self.device = "cuda"
        self.camera_distance = camera_distance
        self.SH = SH()
        self.init_lit = init_lit.reshape([1, 1, -1]).astype(np.float32)

        self.to(self.device)

    def to(self, device):
        self.device = device
        for key, value in self.__dict__.items():
            if type(value).__module__ == np.__name__:
                setattr(self, key, torch.tensor(value).to(device))

    def computer_for_render(self, **coef_dict):
        face_shape = self.compute_shape(coef_dict["id"], coef_dict["exp"])
        rotation = self.compute_rotation(coef_dict["angle"])

        face_shape_transformed = self.transform(
            face_shape, rotation, coef_dict["trans"]
        )
        face_vertex = self.to_camera(face_shape_transformed)

        face_proj = self.to_image(face_vertex)

        face_texture = self.compute_texture(coef_dict["tex"])
        face_norm = self.compute_norm(face_shape)
        face_norm_roted = face_norm @ rotation
        face_color = self.compute_color(
            face_texture, face_norm_roted, coef_dict["gamma"]
        )

        return face_vertex, face_texture, face_color

    def save_mesh(self, name, mask, export=True, **kwargs):
        recon_shape, _, recon_color = self.computer_for_render(**kwargs)
        return self.save_mesh_explict(name, recon_shape, recon_color, mask, export)

    def save_mesh_explict(self, name, recon_shape, recon_color, mask, export=True):
        recon_shape[..., -1] = (
            10 - recon_shape[..., -1]
        )  # from camera space to world space
        recon_shape = recon_shape.detach().cpu().numpy()[0]
        recon_color = recon_color.detach().cpu().numpy()[0]
        if mask == "eye_nose":
            tri = self.face_buf_eye_nose.detach().cpu().numpy()
        elif mask == "eye":
            tri = self.face_buf_eye.detach().cpu().numpy()
        elif mask == "respirator":
            tri = self.face_buf_respirator.detach().cpu().numpy()
        else:
            tri = self.face_buf.detach().cpu().numpy()
        # print('-----before mesh-----')
        mesh = trimesh.Trimesh(
            vertices=recon_shape,
            faces=tri,
            vertex_colors=np.clip(255.0 * recon_color, 0, 255).astype(np.uint8),
            process=False,
        )
        # print('-----after mesh-----')

        if export:
            mesh.export(name)
            return None
        else:
            return mesh

    def forward(self, id, exp, angle, trans, tex, gamma):
        """
        Return:
          face_vertex
          face_color
        Parameters:

        """
        face_shape = self.compute_shape(id, exp)
        rotation = self.compute_rotation(angle)

        face_shape_transformed = self.transform(face_shape, rotation, trans)
        face_vertex = self.to_camera(face_shape_transformed)

        face_texture = self.compute_texture(tex)
        face_norm = self.compute_norm(face_shape)
        face_norm_roted = face_norm @ rotation
        face_color = self.compute_color(face_texture, face_norm_roted, gamma)

        return face_vertex, face_texture, face_color

    def to_camera(self, face_shape):
        face_shape[..., -1] = self.camera_distance - face_shape[..., -1]
        return face_shape

    def to_image(self, face_shape):
        """
        Return:
            face_proj        -- torch.tensor, size (B, N, 2), y direction is opposite to v direction

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """
        # to image_plane
        face_proj = face_shape @ self.persc_proj
        face_proj = face_proj[..., :2] / face_proj[..., 2:]

        return face_proj

    def transform(self, face_shape, rot, trans):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3) pts @ rot + trans

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
            rot              -- torch.tensor, size (B, 3, 3)
            trans            -- torch.tensor, size (B, 3)
        """
        return face_shape @ rot + trans.unsqueeze(1)

    def compute_rotation(self, angles):
        """
        Return:
            rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat

        Parameters:
            angles           -- torch.tensor, size (B, 3), radian
        """

        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1]).to(self.device)
        zeros = torch.zeros([batch_size, 1]).to(self.device)
        x, y, z = (
            angles[:, :1],
            angles[:, 1:2],
            angles[:, 2:],
        )

        rot_x = torch.cat(
            [
                ones,
                zeros,
                zeros,
                zeros,
                torch.cos(x),
                -torch.sin(x),
                zeros,
                torch.sin(x),
                torch.cos(x),
            ],
            dim=1,
        ).reshape([batch_size, 3, 3])

        rot_y = torch.cat(
            [
                torch.cos(y),
                zeros,
                torch.sin(y),
                zeros,
                ones,
                zeros,
                -torch.sin(y),
                zeros,
                torch.cos(y),
            ],
            dim=1,
        ).reshape([batch_size, 3, 3])

        rot_z = torch.cat(
            [
                torch.cos(z),
                -torch.sin(z),
                zeros,
                torch.sin(z),
                torch.cos(z),
                zeros,
                zeros,
                zeros,
                ones,
            ],
            dim=1,
        ).reshape([batch_size, 3, 3])

        rot = rot_z @ rot_y @ rot_x
        return rot.permute(0, 2, 1)

    def compute_texture(self, tex_coeff, normalize=True):
        """
        Return:
            face_texture     -- torch.tensor, size (B, N, 3), in RGB order, range (0, 1.)

        Parameters:
            tex_coeff        -- torch.tensor, size (B, 80)
        """
        batch_size = tex_coeff.shape[0]
        face_texture = (
            torch.einsum("ij,aj->ai", self.tex_base, tex_coeff) + self.mean_tex
        )
        if normalize:
            face_texture = face_texture / 255.0
        return face_texture.reshape([batch_size, -1, 3])

    def compute_norm(self, face_shape):
        """
        Return:
            vertex_norm      -- torch.tensor, size (B, N, 3)

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """

        v1 = face_shape[:, self.face_buf[:, 0]]
        v2 = face_shape[:, self.face_buf[:, 1]]
        v3 = face_shape[:, self.face_buf[:, 2]]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = torch.cross(e1, e2, dim=-1)
        face_norm = F.normalize(face_norm, dim=-1, p=2)
        face_norm = torch.cat(
            [face_norm, torch.zeros(face_norm.shape[0], 1, 3).to(self.device)], dim=1
        )

        vertex_norm = torch.sum(face_norm[:, self.point_buf], dim=2)
        vertex_norm = F.normalize(vertex_norm, dim=-1, p=2)
        return vertex_norm

    def compute_color(self, face_texture, face_norm, gamma):
        """
        Return:
            face_color       -- torch.tensor, size (B, N, 3), range (0, 1.)

        Parameters:
            face_texture     -- torch.tensor, size (B, N, 3), from texture model, range (0, 1.)
            face_norm        -- torch.tensor, size (B, N, 3), rotated face normal
            gamma            -- torch.tensor, size (B, 27), SH coeffs
        """
        batch_size = gamma.shape[0]
        v_num = face_texture.shape[1]
        a, c = self.SH.a, self.SH.c
        gamma = gamma.reshape([batch_size, 3, 9])
        gamma = gamma + self.init_lit
        gamma = gamma.permute(0, 2, 1)
        Y = torch.cat(
            [
                a[0] * c[0] * torch.ones_like(face_norm[..., :1]).to(self.device),
                -a[1] * c[1] * face_norm[..., 1:2],
                a[1] * c[1] * face_norm[..., 2:],
                -a[1] * c[1] * face_norm[..., :1],
                a[2] * c[2] * face_norm[..., :1] * face_norm[..., 1:2],
                -a[2] * c[2] * face_norm[..., 1:2] * face_norm[..., 2:],
                0.5 * a[2] * c[2] / np.sqrt(3.0) * (3 * face_norm[..., 2:] ** 2 - 1),
                -a[2] * c[2] * face_norm[..., :1] * face_norm[..., 2:],
                0.5
                * a[2]
                * c[2]
                * (face_norm[..., :1] ** 2 - face_norm[..., 1:2] ** 2),
            ],
            dim=-1,
        )
        r = Y @ gamma[..., :1]
        g = Y @ gamma[..., 1:2]
        b = Y @ gamma[..., 2:]
        face_color = torch.cat([r, g, b], dim=-1) * face_texture
        return face_color

    def compute_shape(self, id_coeff, exp_coeff):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3)

        Parameters:
            id_coeff         -- torch.tensor, size (B, 80), identity coeffs
            exp_coeff        -- torch.tensor, size (B, 64), expression coeffs
        """
        batch_size = id_coeff.shape[0]
        id_part = torch.einsum("ij,aj->ai", self.id_base, id_coeff)
        exp_part = torch.einsum("ij,aj->ai", self.exp_base, exp_coeff)
        face_shape = id_part + exp_part + self.mean_shape.reshape([1, -1])
        return face_shape.reshape([batch_size, -1, 3])


# ref: https://github.com/sicxu/Deep3DFaceRecon_pytorch/blob/master/models/facerecon_model.py#L17
class Mesh2Image(nn.Module):
    def __init__(self, opt):
        super().__init__()
        fov = 2 * np.arctan(opt.center / opt.focal) * 180 / np.pi
        self.renderer = MeshRenderer(
            rasterize_fov=fov,
            znear=opt.z_near,
            zfar=opt.z_far,
            rasterize_size=int(2 * opt.center),
        )

    def to(self, device):
        self.device = device
        for key, value in self.__dict__.items():
            if type(value).__module__ == np.__name__:
                setattr(self, key, torch.tensor(value).to(device))

    def forward(self, pred_vertex, pred_color, face_buf):
        """
        pred_vertex, pred_color 是 BFM2Mesh 模型的输出
        face_buf 是数据集的 'tri' 数据域
        """
        pred_mask, _, pred_face = self.renderer(pred_vertex, face_buf, feat=pred_color)
        return pred_mask, pred_face


class Pipeline(nn.Module):
    def __init__(
        self,
        opt,
        face_veri_model,
        bfm_coeff_init,
        mask=None,
        pos_factor=0.0,
        bg_factor=0.2,
    ):
        super().__init__()
        self.bfm2mesh = BFM2Mesh(bfm_coeff_init)
        self.mesh2image = Mesh2Image(opt)
        self.face_veri_model = face_veri_model.cuda()
        self.pos_factor = pos_factor
        self.bg_factor = bg_factor
        self.mask = mask

    def render_black(self, url, bg_url=None, **kwargs):
        id = kwargs["id"]
        exp = kwargs["exp"]
        angle = kwargs["angle"]
        trans = kwargs["trans"]
        tex = kwargs["tex"]
        gamma = kwargs["gamma"]

        face_vertex, _, face_color = self.bfm2mesh(id, exp, angle, trans, tex, gamma)

        pred_mask, pred_face = self.mesh2image(
            face_vertex, face_color, self.bfm2mesh.face_buf
        )

        if bg_url == None:
            pred_face = pred_face * 255.0
        elif bg_url == "white":
            img = np.ones((224, 224, 3))
            img_tensor = (
                torch.tensor(img[np.newaxis, :, :, :]).permute(0, 3, 1, 2).to("cuda")
            )
            pred_face = (
                pred_face * 255.0 * pred_mask + (1 - pred_mask) * img_tensor * 255.0
            )
        else:
            bg = imread(bg_url).astype(np.float32)
            h, w, _ = bg.shape
            c = (h // 2, w // 2)

            # print(pred_face.shape)
            _, _, ph, pw = pred_face.shape
            bg = bg[
                np.newaxis,
                c[0] - ph // 2 : c[0] + ph // 2,
                c[1] - pw // 2 : c[1] + pw // 2,
                :,
            ]
            bg = torch.Tensor(bg).permute(0, 3, 1, 2).cuda()
            pred_face = pred_face * 255.0 * pred_mask + (1 - pred_mask) * bg

        imsave(
            url,
            np.clip(
                pred_face.detach().cpu().permute(0, 2, 3, 1).squeeze(0).numpy(), 0, 255
            ).astype(np.uint8),
        )

        return

    def forward(
        self, **kwargs
    ):  # id, exp, angle, trans, tex, gamma, M, align_size, img_tensor
        id = kwargs["id"]
        exp = kwargs["exp"]
        # trans and angle are vector p in original paper
        angle = kwargs["angle"]
        trans = kwargs["trans"]
        tex = kwargs["tex"]
        gamma = kwargs["gamma"]
        M = kwargs["M"]
        align_size = kwargs["align_size"]
        img_tensor = kwargs["img_tensor"]

        if self.pos_factor > 0:
            angle = matrix_diversify(angle, self.pos_factor, "cuda")
            trans = matrix_diversify(trans, self.pos_factor, "cuda")

        face_vertex, _, face_color = self.bfm2mesh(id, exp, angle, trans, tex, gamma)

        if img_tensor is None:
            pred_mask, pred_face = self.mesh2image(
                face_vertex, face_color, self.bfm2mesh.face_buf
            )
            img = np.random.rand(224, 224, 3) * 255 * self.bg_factor
            img_tensor = (
                torch.tensor(img[np.newaxis, :, :, :]).permute(0, 3, 1, 2).to("cuda")
            )
            pred_face = 255 * pred_face  # black backgrounds
        else:
            if self.mask == "T":
                face_buffer = self.bfm2mesh.face_buf_T
            elif self.mask == "hole":
                face_buffer = self.bfm2mesh.face_buf_crop_eye
            elif self.mask == "nose":
                face_buffer = self.bfm2mesh.face_buf_nose
            elif self.mask == "eye":
                face_buffer = self.bfm2mesh.face_buf_eye
            elif self.mask == "mouth":
                face_buffer = self.bfm2mesh.face_buf_mouth
            elif self.mask == "cheek":
                face_buffer = self.bfm2mesh.face_buf_cheek
            elif self.mask == "eye_nose":
                face_buffer = self.bfm2mesh.face_buf_eye_nose
            elif self.mask == "respirator":
                face_buffer = self.bfm2mesh.face_buf_respirator
            else:
                face_buffer = self.bfm2mesh.face_buf

            pred_mask, pred_face = self.mesh2image(face_vertex, face_color, face_buffer)
            # imsave('./mask.png', np.clip((pred_mask[0, :, :, :] * 255.).cpu().detach().numpy().transpose(1, 2, 0), 0, 255).astype(np.uint8))
            # imsave('./face.png', np.clip((pred_face[0, :, :, :] * 255.).cpu().detach().numpy().transpose(1, 2, 0), 0, 255).astype(np.uint8))
            pred_face = (
                255 * pred_face * pred_mask + (1 - pred_mask) * img_tensor
            )  # origin backgrounds

        pred_face = pred_face.float()
        face_input = kornia.warp_affine(pred_face, M, align_size)
        face_embedding = self.face_veri_model.forward(face_input)

        return face_embedding, pred_face

    def get_mask(self, **kwargs):
        id = kwargs["id"]
        exp = kwargs["exp"]
        # trans and angle are vector p in original paper
        angle = kwargs["angle"]
        trans = kwargs["trans"]
        tex = kwargs["tex"]
        gamma = kwargs["gamma"]

        if self.mask == "T":
            face_buffer = self.bfm2mesh.face_buf_T
        elif self.mask == "hole":
            face_buffer = self.bfm2mesh.face_buf_crop_eye
        elif self.mask == "nose":
            face_buffer = self.bfm2mesh.face_buf_nose
        elif self.mask == "eye":
            face_buffer = self.bfm2mesh.face_buf_eye
        elif self.mask == "mouth":
            face_buffer = self.bfm2mesh.face_buf_mouth
        elif self.mask == "cheek":
            face_buffer = self.bfm2mesh.face_buf_cheek
        elif self.mask == "eye_nose":
            face_buffer = self.bfm2mesh.face_buf_eye_nose
        elif self.mask == "respirator":
            face_buffer = self.bfm2mesh.face_buf_respirator
        else:
            face_buffer = self.bfm2mesh.face_buf
        face_vertex, _, face_color = self.bfm2mesh(id, exp, angle, trans, tex, gamma)
        pred_mask, pred_face = self.mesh2image(face_vertex, face_color, face_buffer)
        return pred_mask, pred_face


class Baseline(nn.Module):
    def __init__(self, opt, face_veri_model, bfm_coeff_init, mask=None, bg_factor=0.2):
        super().__init__()
        self.bfm2mesh = BFM2Mesh(bfm_coeff_init)
        self.mesh2image = Mesh2Image(opt)
        self.face_veri_model = face_veri_model.cuda()
        self.bg_factor = bg_factor
        self.mask = mask

    def forward(self, **kwargs):
        face_vertex = kwargs["vertex"]
        face_color = kwargs["color"]
        M = kwargs["M"]
        align_size = kwargs["align_size"]
        img_tensor = kwargs["img_tensor"]

        if img_tensor is None:
            pred_mask, pred_face = self.mesh2image(
                face_vertex, face_color, self.bfm2mesh.face_buf
            )
            img = np.random.rand(224, 224, 3) * 255 * self.bg_factor
            img_tensor = (
                torch.tensor(img[np.newaxis, :, :, :]).permute(0, 3, 1, 2).to("cuda")
            )
            pred_face = 255 * pred_face  # black backgrounds
        else:
            if self.mask == "T":
                face_buffer = self.bfm2mesh.face_buf_T
            elif self.mask == "hole":
                face_buffer = self.bfm2mesh.face_buf_crop_eye
            elif self.mask == "nose":
                face_buffer = self.bfm2mesh.face_buf_nose
            elif self.mask == "eye":
                face_buffer = self.bfm2mesh.face_buf_eye
            elif self.mask == "mouth":
                face_buffer = self.bfm2mesh.face_buf_mouth
            elif self.mask == "cheek":
                face_buffer = self.bfm2mesh.face_buf_cheek
            elif self.mask == "eye_nose":
                face_buffer = self.bfm2mesh.face_buf_eye_nose
            elif self.mask == "respirator":
                face_buffer = self.bfm2mesh.face_buf_respirator
            else:
                face_buffer = self.bfm2mesh.face_buf

            pred_mask, pred_face = self.mesh2image(face_vertex, face_color, face_buffer)

            pred_face = (
                255 * pred_face * pred_mask + (1 - pred_mask) * img_tensor
            )  # origin backgrounds

        pred_face = pred_face.float()
        face_input = kornia.warp_affine(pred_face, M, align_size)
        face_embedding = self.face_veri_model.forward(face_input)

        return face_embedding, pred_face
