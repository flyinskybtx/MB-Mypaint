import random
from pathlib import Path
import os.path as osp
import numpy as np

import tensorflow.keras as keras

from Data.gen_data import gen_samples
from Data.load_HWDB import translate_stroke , centralize_char, interpolate_stroke
from utils.mypaint_agent import MypaintAgent

HWDB_DIR = Path(__file__).parent / 'pot'


class CnpDataGenerator(keras.utils.Sequence):

    def __init__(self, authors, image_size=192, roi_size=16):
        self.authors = authors
        self.image_size = image_size
        self.roi_size = roi_size
        self.cur_author = None
        self.strokes = None
        self.agent = MypaintAgent({'brush_name': 'custom/slow_ink'})

        self.update_samples()

    def __len__(self):
        num_batches = len(self.strokes)
        return num_batches

    def __getitem__(self, index):
        stroke = random.choice(self.strokes)
        samples = gen_samples(stroke, self.agent)

        X_disp = np.array([s[:3] for s in all_samples])
        X_img = np.array([s[3] for s in all_samples])
        Y = np.array([s[4] for s in all_samples])

        return[X_img, X_disp], Y

    def on_epoch_end(self):
        self.update_samples()

    def update_samples(self):
        self.cur_author = random.choice(self.authors)
        pot_file = osp.join(HWDB_DIR, f'{self.cur_author}.pot')
        words = translate_pot(pot_file)
        centralized_words = [centralize_strokes(word, dst_size=(self.image_size, self.image_size)) for word in
                             words.values()]
        strokes = [s for w in centralized_words for s in w]
        self.strokes = [interpolate_stroke(s, self.roi_size / 2) for s in strokes]


if __name__ == '__main__':
    data_generator = CnpDataGenerator(['1001-c', '1002-c'])
    print(data_generator.__getitem__(1))
