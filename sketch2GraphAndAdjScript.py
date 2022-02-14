from utils.sketch_processing import draw_three
from utils.data_process import get_node_coordinates_graph

import numpy as np
import os
from tqdm import tqdm

# split_nums, node_nums = [int(x) for x in outPath.replace("/", "").split("_")[-2:]]
outPath = "./dataset_32_150"
split_nums = 32
node_nums = 150
mode = "train"

def make_coordinate_graph(sketch: np.ndarray, mask_prob: float, dThreshold: float = 0.2):
    canvas = draw_three(sketch, img_size=256)
    result_points, A = get_node_coordinates_graph(canvas, split_nums, split_nums,
                                                  maxPointFilled=node_nums,
                                                  mask_prob=mask_prob, max_pixel_value=256 - 1,
                                                  dThreshold=dThreshold)
    return result_points, A

def generator(in_sketches: np.ndarray, out_path: str, name: str, dThreshold: float = 0.2):
    nodes = []
    adjs = []
    for sketch in tqdm(in_sketches):
        n, a = make_coordinate_graph(sketch, mask_prob=0.0, dThreshold=dThreshold)
        nodes.append(np.expand_dims(n, 0).astype("uint8"))
        a = a * 100
        adjs.append(np.expand_dims(a, 0).astype("uint8"))
    nodes = np.concatenate(nodes).astype("uint8")
    adjs = np.concatenate(adjs).astype("uint8")

    np.savez(f"{out_path}/{name}_nodes_{mode}.npz", **{mode: nodes})
    np.savez(f"{out_path}/{name}_adjs_{mode}.npz", **{mode: adjs})
    print("success save", f"{out_path}/{name}_nodes_{mode}.npz", f"{out_path}/{name}_adjs_{mode}.npz")


if __name__ == '__main__':
    import shutil

    os.makedirs(f"./{outPath}", exist_ok=True)
    data_location = './dataset'

    data_list = [
        "airplane.npz",
        "angel.npz",
    ] # test 2

    for each in data_list:
        print(each)
        shutil.copy(f"{data_location}/{each}", f"{outPath}/{each}")
        sketches = np.load(f"{data_location}/{each}", allow_pickle=True, encoding="latin1")[mode]
        generator(sketches, outPath, name=each, dThreshold=0.2)
