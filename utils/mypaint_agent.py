import os
import pathlib
import random
import string
import time

import cv2
import numpy as np

import sys

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

MY_BRUSH_PATH = os.path.join(cur_dir.parent, 'Brushes')
BRUSH_PATH = mypaint_brushdir
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


class MypaintAgent:
    def __init__(self, env_config: dict, num_tiles=None):
        """

        :param env_config:
        """
        brush_name = env_config.setdefault('brush_name', 'classic/calligraphy')
        dtime = env_config.setdefault('dtime', 0.05)

        if os.path.exists(os.path.join(MY_BRUSH_PATH, f'{brush_name}.myb')):
            self.brush_path = os.path.join(MY_BRUSH_PATH, f'{brush_name}.myb')
        else:
            self.brush_path = os.path.join(BRUSH_PATH, f'{brush_name}.myb')

        self.dtime = dtime

        if num_tiles is None:
            self.num_tiles = DICT_NUM_TILES[brush_name]
        else:
            self.num_tiles = num_tiles

        self.brush = None
        self.surface = None
        self.step = 0
        self.time = 0

        self.helper_brush = self.get_brush(os.path.join(BRUSH_PATH, 'classic/pen.myb'))
        self.reset()

    @staticmethod
    def get_brush(brush_path):
        with open(brush_path) as fp:
            return mypaint_brush.Brush(mypaint_brush.BrushInfo(fp.read()))

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

        self.brush = self.get_brush(self.brush_path)
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

    def get_img(self, tar_shape=None):
        """ Get current image in Mypaint, and resize to target shape """
        # Save temp image
        filename = os.path.join(INTERMEDIATE_IMG_SAVE_DIR, ''.join(random.choices(string.ascii_uppercase +
                                                                                  string.digits, k=12)))
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

        if tar_shape:
            img = cv2.resize(img, tar_shape)
            img[np.where(img > 0)] = 1
        os.remove(filename)

        return img


if __name__ == '__main__':
    agent = MypaintAgent({'brush_name': 'custom/slow_ink'})
    agent.reset()
    # agent.paint(0, 0, 1)
    agent.paint(0.5, 0.5, 0)
    agent.paint(0.5, 0.5, 1)
    agent.paint(0.8, 0.8, 1)
    agent.paint(0.2, 0.8, 1)
    agent.paint(0.8, 0.2, 1)

    result = agent.get_img()
    print(result)

    import matplotlib.pyplot as plt

    plt.imshow(result)
    plt.show()
