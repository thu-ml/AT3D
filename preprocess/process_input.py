from __future__ import absolute_import

import os
import sys

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AT3D.utils import save_align_matrix, save_embedding


def process_input(img_dir):
    if not os.path.exists(img_dir):
        print("The directory does not exist.")
        return
    for subdir in tqdm(os.listdir(img_dir)):
        dir = os.path.join(img_dir, subdir)
        if os.path.isdir(dir):
            save_align_matrix(
                dir=dir, save_prefix=dir, image_shapes=[(112, 112), (112, 96)]
            )
            save_embedding(
                dir=dir, save_prefix=dir, model_names=["ArcFace", "CosFace", "ResNet50"]
            )


if __name__ == "__main__":
    workdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    process_input(img_dir=f"{workdir}/data/demo/origin")
