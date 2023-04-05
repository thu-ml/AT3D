import os
from multiprocessing.dummy import Pool

import cv2
from mtcnn import MTCNN


# https://github.com/ipazc/mtcnn
def single_mtcnn(path):
    if not os.path.exists(path):
        return
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    dic = detector.detect_faces(img)
    print(path, flush=True)

    if len(dic) == 0:
        print(f"{path} error!")
        return
    dic.sort(key=lambda x: abs((x["box"][0] + x["box"][2] // 2) - img.shape[1] // 2))
    filename = path.split(os.path.sep)[-1].replace(".jpg", "").replace(".png", "")
    with open(
        os.path.join(os.path.dirname(path), f"detection_{filename}.txt"),
        "w",
    ) as f:
        for _, v in dic[0]["keypoints"].items():
            f.write("{}\t{}\n".format(v[0], v[1]))


def main():
    pool = Pool(processes=16)
    dataset = "demo"
    workdir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    )
    print(workdir)
    files = []
    for subdir in os.listdir(os.path.join(workdir, "data", dataset, "origin")):
        subdir = os.path.join(workdir, "data", dataset, "origin", subdir)
        if os.path.isdir(subdir):
            for file in os.listdir(subdir):
                if (
                    file.endswith(".jpg")
                    or file.endswith(".png")
                    and (not file.startswith("final_"))
                ):
                    files.append(os.path.join(subdir, file))
    print(files)

    pool.map(single_mtcnn, files)
    pool.close()


if __name__ == "__main__":
    main()
