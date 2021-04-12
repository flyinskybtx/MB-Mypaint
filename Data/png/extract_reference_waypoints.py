import os
import pickle

import tqdm

from Data import DATA_DIR
from Data.data_process_lib import load_stroke_png, preprocess_stroke_png, img_to_skeleton_path, skeleton_path_to_wps
from script.main_procs.hparams import define_hparams
import numpy as np
import matplotlib.pyplot as plt


def save_skeleton_waypoints(config=None):
    if config is None:
        config = define_hparams()
    # --- Load images, preprocess, extract reference path, save to json
    data = {}
    for num in tqdm.trange(64):
        img = load_stroke_png(num)
        img = preprocess_stroke_png(img, image_size=config.image_size)
        skeleton_path = img_to_skeleton_path(img)
        data[num] = skeleton_path

        frame = np.zeros_like(img)
        for line in skeleton_path:
            for point in line:
                frame[point[0] - 5:point[0] + 5, point[1] - 5:point[1] + 5] = 1
        plt.imshow(np.stack([frame, img, np.zeros_like(img)], axis=-1))
        plt.show()
        plt.suptitle(f'Img.{num}')
    with open(os.path.join(DATA_DIR, f'png/skeleton_paths_{config.image_size}.pkl'), 'wb') as fp:
        pickle.dump(data, fp)


def load_all_png_skeleton_waypoints(image_size=None):
    if image_size is None:
        config = define_hparams()
        image_size = config.image_size

    with open(os.path.join(DATA_DIR, f'png/skeleton_paths_{image_size}.pkl'), 'rb') as fp:
        data = pickle.load(fp)
        return data


if __name__ == '__main__':
    # save_skeleton_waypoints()

    config = define_hparams()
    data = load_all_png_skeleton_waypoints()

    for k, skeleton_path in data.items():
        img = load_stroke_png(k)
        img = preprocess_stroke_png(img, image_size=config.image_size)
        frame = np.zeros_like(img)
        for line in skeleton_path:
            for point in line:
                frame[point[0] - 5:point[0] + 5, point[1] - 5:point[1] + 5] = 1
        plt.imshow(np.stack([frame, img, np.zeros_like(img)], axis=-1))
        plt.show()
        plt.suptitle(f'Img.{k}')
