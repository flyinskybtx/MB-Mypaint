from datetime import datetime

import matplotlib.pyplot as plt
from glob import glob
import os
import os.path as osp
import pathlib
import random
import string
import sys
import time

import cv2
import numpy as np

from Data import DATA_DIR
from Model import AttrDict
from script.main_procs.hparams import define_hparams

try:
    from lib import mypaintlib, tiledsurface
    from lib.config import mypaint_brushdir
    import lib.brush as mypaint_brush
except ModuleNotFoundError:
    sys.path.append('/home/flyinsky/.local/lib/mypaint')
    from lib import mypaintlib, tiledsurface
    from lib.config import mypaint_brushdir
    import lib.brush as mypaint_brush

np.set_printoptions(formatter={'float_kind': lambda x: "%.2f" % x})
cur_dir = pathlib.Path(__file__)

BRUSH_DIR = os.path.join(cur_dir.parent, 'Brushes')
INTERMEDIATE_IMG_SAVE_DIR = os.path.join(cur_dir.parent.parent, 'Data/temp_imgs')

TILE_SIZE = mypaintlib.TILE_SIZE
REAL_IMG_DIR = os.path.join(pathlib.Path(__file__).parent.parent, 'Data/png')

DICT_NUM_TILES = {
    'custom/slow_ink': 2.0,
    'custom/brush': 1.6,
    'classic/brush': 3.8,
    'classic/pen': 1.7,
    'classic/calligraphy': 3.0,
    'custom/ink': 2.0,

}


def get_brushinfo_file(brush_name, agent_name):
    filename = os.path.join(DATA_DIR, 'offline', brush_name.split('/')[-1], agent_name, 'BrushInfo.myb')
    return filename


class MypaintPainter:
    def __init__(self, brush_config, num_tiles=None):
        """

        :param brush_config:
        """

        self.brush_name = brush_config.brush_name
        self.brush_info_file = get_brushinfo_file(brush_config.brush_name, brush_config.agent_name)
        self.dtime = brush_config.setdefault('dtime', 0.05)

        if num_tiles is None:
            self.num_tiles = DICT_NUM_TILES[self.brush_name]
        else:
            self.num_tiles = num_tiles

        self.brush = None
        self.surface = None
        self.step = 0
        self.time = 0

        helper_brush_info = mypaint_brush.BrushInfo()
        with open(os.path.join(mypaint_brushdir, 'classic/pen.myb'), 'r') as fp:
            helper_brush_info.from_json(fp.read())
        self.helper_brush = mypaint_brush.Brush(helper_brush_info)
        self.reset()

    @staticmethod
    def get_brush(brush_file):
        brush_info = mypaint_brush.BrushInfo()
        with open(brush_file, 'r') as fp:
            brush_info.from_json(fp.read())
        return mypaint_brush.Brush(brush_info)

    def paint(self, x: float, y: float, pressure, x_tilt=0, y_tilt=0, view_zoom=1, view_rotation=0, barrel_rotation=1):
        """ paint a step

        :param x:
        :param y:
        :param pressure:
        :param x_tilt:
        :param y_tilt:
        :param view_zoom:
        :param view_rotation:
        :param barrel_rotation:
        :return:
        """

        assert 0 <= x <= 1 and 0 <= y <= 1

        self.step += 1
        self.time += self.dtime

        x = int(x * self.num_tiles * TILE_SIZE + TILE_SIZE)
        y = int(y * self.num_tiles * TILE_SIZE + TILE_SIZE)
        pressure = float(pressure)

        self.surface.begin_atomic()
        self.brush.stroke_to(
            self.surface.backend, x, y, pressure,
            x_tilt, y_tilt, self.time, view_zoom, view_rotation, barrel_rotation)
        self.surface.end_atomic()

    def _border_paint(self, x, y, pressure):
        self.time += 0.1

        self.surface.begin_atomic()
        self.helper_brush.stroke_to(
            self.surface.backend, x, y, pressure, 0, 0, self.time, 1, 0, 1)
        self.surface.end_atomic()

    def reset(self):
        if self.surface:
            del self.surface

        self.time = 0

        self.brush = self.get_brush(self.brush_info_file)
        self.surface = tiledsurface.Surface()

        # Draw box, add TILESIZE DISPLACEMENT
        self._border_paint(0, 0, 0)
        self._border_paint(0, 0, 1)
        self._border_paint(TILE_SIZE * (np.ceil(self.num_tiles) + 2), 0, 1)
        self._border_paint(TILE_SIZE * (np.ceil(self.num_tiles) + 2),
                           TILE_SIZE * (np.ceil(self.num_tiles) + 2), 1)
        self._border_paint(0, TILE_SIZE * (np.ceil(self.num_tiles) + 2), 1)
        self._border_paint(0, 0, 1)
        self._border_paint(0, 0, 0)

        self.step = 0

    def get_img(self, shape=None):
        """ Get current image in Mypaint, and resize to target shape """
        # Save temp image
        filename = os.path.join(INTERMEDIATE_IMG_SAVE_DIR, str(time.time())+''.join(random.choices(
            string.ascii_uppercase, k=10)))
        self.surface.save_as_png(filename, alpha=True)
        time.sleep(1e-5)

        # Load temp image
        try:
            img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            # Select one channel
            img = img[:, :, -1]
        except TypeError:
            return self.get_img()

        # Clip image to drawing size
        img = img[
              2 * TILE_SIZE: int((self.num_tiles + 2) * TILE_SIZE),
              2 * TILE_SIZE: int((self.num_tiles + 2) * TILE_SIZE)
              ]
        img = img.transpose()

        if shape:
            img = cv2.resize(img, shape)
            # img[np.where(img > 0)] = 1
        os.remove(filename)

        return img


def test_standard_move(brush_config: AttrDict):
    agent = MypaintPainter(brush_config)
    agent.reset()
    # mypaint_painter.paint(0, 0, 1)
    agent.paint(0.5, 0.5, 0)
    agent.paint(0.5, 0.5, 1)
    agent.paint(0.8, 0.8, 1)
    agent.paint(0.2, 0.8, 1)
    agent.paint(0.8, 0.2, 1)

    img = agent.get_img()
    return img


if __name__ == '__main__':
    config = AttrDict()
    config.brush_name = 'custom/slow_ink'
    config.agent_name = 'Physics'

    img = test_standard_move(config)
    plt.imshow(img)
    plt.show()
