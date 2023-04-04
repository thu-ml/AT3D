from .base_loader import BaseLoader
from .patch_loader import PatchLoader, PatchLoader2D
from .utils import read_3dmm_mat, read_align_mat, read_feat

LOADER_DICT = {
    "whole": PatchLoader,
    "patch": PatchLoader,
    "patch_2d": PatchLoader2D,
}
