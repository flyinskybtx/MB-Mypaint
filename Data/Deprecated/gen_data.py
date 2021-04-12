import glob
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from Data.HWDB.load_HWDB import interpolate_stroke
from utils.mypaint_agent import MypaintPainter

HWDB_DIR = Path(__file__).parent / 'pot'
IMAGE_SIZE = 192
ROI_SIZE = 16


def gen_samples(stroke, agent, image_size=IMAGE_SIZE, margin=ROI_SIZE):
    # Initialize
    agent.reset()
    stroke = stroke.astype(np.float)
    x0, y0 = stroke[0]
    z0 = 0
    agent.paint(x0 / image_size, y0 / image_size, z0)
    prev_img = agent.get_img(shape=(image_size, image_size))
    samples = []

    # Random Z
    z = random.random()

    # Iterate
    for p in stroke:
        # Prev
        prev = cut_roi(prev_img, x0, y0, margin=margin)

        # Step
        x, y = p[:2]
        z = np.clip(z + random.sample([-0.1] * 3 + [0] * 3 + [0.1] * 4, 1)[0], 0, 1)
        agent.paint(x / image_size, y / image_size, z)
        img = agent.get_img(shape=(image_size, image_size))
        post = cut_roi(img - prev_img, x0, y0, margin=margin)

        # Discard empty
        if np.sum(img) > 0 or np.sum(post) > 0:
            # update
            samples.append(((x - x0) / margin,
                            (y - y0) / margin,
                            (z - z0) / margin,
                            np.expand_dims(prev, axis=-1),
                            np.expand_dims(post, axis=-1),
                            ))
        prev_img, x0, y0, z0 = img, x, y, z

    return samples


def gen_cnp_samples(stroke, agent, image_size=IMAGE_SIZE, margin=ROI_SIZE):
    # Initialize
    agent.reset()
    stroke = stroke.astype(np.float)
    x0, y0 = stroke[0]
    z0 = 0
    agent.paint(x0 / image_size, y0 / image_size, z0)
    prev_img = agent.get_img(shape=(image_size, image_size))
    samples = []

    # Random Z
    z = random.random()

    # Iterate
    for p in stroke:
        # Prev
        prev = cut_roi(prev_img, x0, y0, margin=margin)

        # Step
        x, y = p[:2]
        z = np.clip(z + random.sample([-0.1] * 3 + [0] * 3 + [0.1] * 4, 1)[0], 0, 1)
        agent.paint(x / image_size, y / image_size, z)
        img = agent.get_img(shape=(image_size, image_size))
        delta = img - prev_img
        if np.sum(delta) > 0:
            ret, thresh = cv2.threshold(delta.astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img, contour, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            ellipse = cv2.fitEllipse(contour[0])
            center_y, center_x = int(ellipse[0][0]), int(ellipse[0][1])
            axis_y, axis_x = ellipse[1][0], ellipse[1][1]
            rot = ellipse[2]
        else:
            center_x, center_y, axis_x, axis_y, rot = x, y, 0, 0, 0

        samples.append((x - center_x, y - center_y,))

        # Discard empty
        if np.sum(img) > 0 or np.sum(post) > 0:
            # update
            samples.append(((x - x0) / margin,
                            (y - y0) / margin,
                            (z - z0) / margin,
                            np.expand_dims(prev, axis=-1),
                            np.expand_dims(post, axis=-1),
                            ))
        prev_img, x0, y0, z0 = img, x, y, z

    return samples


def cut_roi(img, x, y, margin):
    img = np.pad(img, (margin, margin), 'constant')
    return img[int(x): int(x) + 2 * margin + 1, int(y): int(y) + 2 * margin + 1]


if __name__ == '__main__':
    agent = MypaintPainter({'brush_name': 'custom/slow_ink'})

    for pot_file in sorted(glob.glob(f'{HWDB_DIR}/*.pot'))[1:2]:
        results = translate_pot(pot_file)
        centralized_words = [centralize_strokes(word, dst_size=(IMAGE_SIZE, IMAGE_SIZE)) for word in results.values()]
        strokes = [s for w in centralized_words for s in w]
        strokes = [interpolate_stroke(s, ROI_SIZE / 2) for s in strokes]

        for i, stroke in enumerate(strokes):
            samples = gen_samples(stroke, agent)
            print(f'{i}/{len(strokes)}: len{len(samples)} ')

            for s in samples:
                print(s[:3])
                frame = np.zeros((2 * ROI_SIZE + 1, 2 * ROI_SIZE + 1, 3))
                # frames[:, :, 0:1] = s[3]
                # plt.imshow(frames)
                # plt.show()
                frame[:, :, 1:2] = s[4]
                plt.imshow(frame)
                plt.show()
        break
