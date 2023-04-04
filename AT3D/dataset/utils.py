import numpy as np
import torch
from scipy.io import loadmat


def read_align_mat(path, device, keyword="matrix"):
    M = np.load(path, allow_pickle=True)[keyword]
    if M.shape == (2, 3):
        return torch.Tensor(M[np.newaxis, :, :]).to(device)
    else:
        return torch.zeros(1, 2, 3).to(device)


def read_feat(path, modelname, device):
    return torch.Tensor(np.load(path)[modelname]).to(device)


def read_3dmm_mat(
    path,
    device,
    need_attack=["id", "exp", "tex"],
    return_coeff=["id", "tex", "exp", "angle", "trans", "gamma"],
):
    return_dict = dict()
    coeff_dict = loadmat(path)

    for key in return_coeff:
        coeff = torch.from_numpy(coeff_dict[key]).to(device)
        coeff.requires_grad = True if key in need_attack else False
        return_dict[key] = coeff

    return return_dict
