import json
import os.path as osp

import numpy as np
from tqdm import tqdm

from Data import DATA_DIR
from Data.data_process_lib import extract_skeleton_trace, load_stroke_png, preprocess_stroke_png, refpath_to_actions, \
    load_imgs_and_refpaths
from script.main_procs.hparams import define_hparams

if __name__ == '__main__':
    config = define_hparams()
    # --- Load images, preprocess, extract reference path, save to json
    imgs = [load_stroke_png(num) for num in range(64)]
    imgs = [preprocess_stroke_png(img, image_size=config.image_size) for img in imgs]
    paths = []
    for img in tqdm(imgs):
        ref_path = extract_skeleton_trace(img, config.xy_grid, display=True)
        paths.append(ref_path)
        ref_actions = refpath_to_actions(ref_path,
                                         config.xy_grid,
                                         config.action_shape)
        for action in ref_actions:
            assert np.min(action) >= 0 and np.max(action) < config.action_shape

    json_data = {num: [img.tolist(), path.tolist()] for num, (img, path) in enumerate(zip(imgs, paths))}
    with open(osp.join(DATA_DIR, 'png', 'ref_paths.json'), 'w') as fp:
        json.dump(json_data, fp)

    load_imgs_and_refpaths()
