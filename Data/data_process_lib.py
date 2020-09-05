import glob
import itertools
import os.path as osp
import random
import struct
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from scipy import ndimage
from skimage.draw import line, line_aa
from skimage.morphology import skeletonize

from utils.mypaint_agent import MypaintAgent

HWDB_DIR = '/home/baitianxiang/Workspace/MB-Mypaint/Data/pot'
if not '/home/baitianxiang/Workspace/MB-Mypaint' in sys.path:
    sys.path.extend(['/home/baitianxiang/Workspace/MB-Mypaint'])


def view_all_png():
    for filename in sorted(glob.glob('./png/*.png')):
        img = plt.imread(filename)
        print(f"IMG: {filename.strip('.png').split('/')[-1]}")
        plt.imshow(img)
        plt.show()


def load_stroke_png(stroke_num):
    filename = osp.join('../Data/png', f'{stroke_num}.png')

    img = plt.imread(filename)[:, :, 0]
    return img


def preprocess_stroke_png(img, image_size, angle=0):
    """ Reshape, Rotate and Binaries """
    from scipy import ndimage
    img = img.copy()
    img = ndimage.rotate(img, angle, reshape=False)
    img[img < 0.1] = 0

    img = cv2.resize(img, (image_size, image_size))
    img[np.where(img > 0.5)] = 1.0
    img[np.where(img <= 0.5)] = 0
    return img


def find_low_resolution_endpoints(img, new_size, return_skel=False):
    """ Find endpoints in low resolution image, to help choose better endpoints """
    old_size = img.shape[0]
    skel_img = cv2.resize(img, (new_size, new_size))
    skel_img = np.round(skel_img)

    skel = skeletonize(skel_img)
    skel = skel.astype(np.float32)

    kernel = np.float32([[1, 1, 1], [1, 10, 1], [1, 1, 1]])

    output = ndimage.convolve(skel, kernel, mode='constant', cval=0.0)
    rr, cc = np.where(output == 11)

    endpoints = [(x, y) for x, y in zip(rr, cc)]
    endpoints = sorted(endpoints, key=lambda x: x[0]**2 + x[1]**2)  # top-left for 1st point
    endpoints = (np.array(endpoints) * old_size / new_size).astype(np.int)

    if return_skel:
        return endpoints, skel
    else:
        return endpoints


def extract_skeleton_trace(img, step_size, discrete=False, display=False):
    """
    从图像中提取笔顺骨架，提取 N 个2维路径点，点间距由step_size决定, 横、纵两方向的移动量小于step_size
    :param img:
    :param step_size:
    :return:
    """
    endpoints, skel = find_low_resolution_endpoints(img, img.shape[0], return_skel=True)
    if len(endpoints) > 3:
        low_size = int(img.shape[0] * 0.8)
        while low_size > 0.4 * img.shape[0]:
            low_endpoints = find_low_resolution_endpoints(img, low_size)
            if 2 <= len(low_endpoints) <= 3:
                break
            low_size = int(low_size * 0.8)

        candidates = []
        for le in low_endpoints:
            candidates.append(sorted(endpoints, key=lambda x: np.sum(np.square(x-le)))[0])

        endpoints = candidates

    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    path = []

    for s, e in zip(endpoints, endpoints[1:]):
        path += [s] * (step_size + 1)
        grid = Grid(matrix=skel.T)
        path += finder.find_path(grid.node(*s), grid.node(*e), grid)[0][1:-1]
    path += [endpoints[-1]] * (step_size + 1)
    path = path[0::step_size + 1]
    path = np.array(path)

    if display:
        rr, cc = np.split(path, 2, axis=-1)

        ep_frame = np.zeros(img.shape)
        ep_frame[rr, cc] = 1

        kernel = np.ones((step_size, step_size))
        ep_frame = ndimage.convolve(ep_frame,
                                    kernel,
                                    mode='constant',
                                    cval=0.0)

        plt.imshow(np.stack([ep_frame, img, np.zeros(img.shape)], axis=-1))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    if not discrete:
        return path
    else:
        path = path / np.array([step_size, step_size])
        path = path.astype(np.int)
        return path


def read_pot(author):
    """Read *.pot file, return {word:stroke}"""
    pot_file = osp.join(HWDB_DIR, f'{author}.pot')

    with open(pot_file, 'rb') as fp:
        contents = fp.read()
        contents = contents.split(b'\xff\xff\xff\xff')[:-1]
        # translate content
        results = {}
        for content in contents:
            sample_size = struct.unpack('H', content[:2])[0]
            assert sample_size == len(content) + 4, f'{sample_size}, {len(content)}'
            word = content[2:6]
            word = word[1::-1].decode('GBK')
            strokes = content[8:]
            strokes = strokes.split(b'\xff\xff\x00\x00')[:-1]
            num_strokes = struct.unpack('H', content[6:8])[0]
            assert num_strokes == len(strokes), 'number of strokes no match'
            strokes = [translate_stroke(s) for s in strokes]
            results[word] = strokes
    print('Finish translating {}'.format(author))
    return results


def translate_stroke(content):
    """ Unpack a stroke encoding to a list of (X,Y) waypoints"""

    points = [content[p:p + 4] for p in range(0, len(content), 4)]
    # Switch x and y
    return np.array([[struct.unpack('h', p[2:])[0], struct.unpack('h', p[:2])[0]] for p in points])


def centralize_char(char_strokes, dst_size, margin=0):
    mins = np.min(np.concatenate(char_strokes, axis=0), axis=0)
    char_strokes = [s - mins for s in char_strokes]  # Cut to bottom-left

    cur_size = np.max(np.concatenate(char_strokes, axis=0), axis=0)
    scale = np.array([dst_size - 2 * margin, dst_size - 2 * margin]) / cur_size  # Calc scale to dst size

    char_strokes = [np.round(s * np.min(scale)) for s in char_strokes]
    char_strokes = [np.array([list(g)[0] for k, g in itertools.groupby(s, lambda x: str(x))]) for s in
                    char_strokes]  # Remove duplicate

    disp = np.round((np.array([dst_size, dst_size]) - np.max(np.concatenate(char_strokes, axis=0), axis=0)) / 2)
    char_strokes = [s + disp for s in char_strokes]  # Move to center
    char_strokes = [s.astype(np.int) for s in char_strokes]
    return char_strokes


def interpolate_stroke(stroke, interval=2):
    stroke = stroke.astype(np.float)
    new_stroke = []
    for p1, p2 in zip(stroke, stroke[1:]):
        num_p = int(max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1])) / interval)
        for i in range(num_p):
            new_stroke.append(p1 + i * (p2 - p1) / num_p)

    new_stroke.append(stroke[-1])
    return np.array(new_stroke, np.int)


def vis_char(char_strokes, dst_size):
    frame = np.zeros((dst_size, dst_size))

    for stroke in char_strokes:
        for p0, p1 in zip(stroke, stroke[1:]):
            rr, cc = line(*p0, *p1)
            frame[rr, cc] = 1
    plt.imshow(frame)
    plt.show()


def get_ellipse(img):
    """
    Get ellipse from incremental image
    return: y, x, axis_y, axis_x, rotation
    """
    if np.sum(img) > 0:
        ret, thresh = cv2.threshold(img.astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contour, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        ellipse = ((0, 0), (0, 0), 0)  # In case ellipse not found
        for i, ct in enumerate(contour):
            if i >= 1:
                break
            if len(ct) >= 5:
                try:
                    ellipse = cv2.fitEllipse(ct)
                except:
                    print(ct)
                    print(contour)
                break
    else:
        ellipse = ((0, 0), (0, 0), 0)
    return ellipse


def stroke_to_trace(stroke: np.ndarray, agent, img_size: int, roi_size: int, display=False):
    """
    Paint stroke, return painted trace.
    return: {'u':control, 's':state, 's0':start point}, where
        control = [dx/roi, dy/roi, dz]
        state = [ellipse refer to holdpos], as [dy/roi,dx/roi,ry/roi,rx/roi,angle/180]
    """
    # Initialize start position.
    stroke = stroke.astype(np.float)
    x0, y0 = stroke[0][:2]
    z0 = round(random.random(), 1)
    img0 = np.zeros((img_size, img_size))
    trace = {'u': [], 's': [], 's0': np.array([x0, y0, z0])}

    agent.reset()
    agent.paint(x0 / img_size, y0 / img_size, 0)
    frame = np.zeros((img_size, img_size, 3))

    # Iterate over stroke waypoints
    for wp in stroke:
        x1, y1 = wp[:2]
        z1 = np.clip(
            z0 + random.sample([0.1] * 4 + [0] * 3 + [-0.1] * 3, 1)[0], 0, 1)
        control = np.array([(x1 - x0) / roi_size,
                            (y1 - y0) / roi_size, z1 - z0])

        # Agent step
        agent.paint(x1 / img_size, y1 / img_size, z1)
        img1 = agent.get_img(tar_shape=(img_size, img_size))  # Should I cut to roi? img1 = cut_roi(img1, x1, y1,
        # roi_size)

        ellipse = get_ellipse(img1 - img0)
        if ellipse[0] == (0, 0):  # Move ellipse to center at x0, y0
            ellipse = ((y0, x0), (0, 0), 0)

        state = np.concatenate([(np.array(ellipse[0]) - np.array([y0, x0])) / roi_size,
                                np.array(ellipse[1]) / roi_size,
                                np.array([ellipse[2] / 180])],
                               axis=0)

        trace['u'].append(control)
        trace['s'].append(state)

        if display:
            frame[:, :, 0] = np.zeros((img_size, img_size))
            frame = cv2.ellipse(frame, ellipse, color=(0, 0, 1), thickness=2)
            frame = cv2.rectangle(
                frame, (int(y1) - roi_size, int(x1) - roi_size),
                (int(y1) + roi_size + 1, int(x1) + roi_size + 1),
                (1, 0, 0), 4)
            rr, cc, val = line_aa(int(x0), int(y0), int(x1), int(y1))
            frame[rr, cc, 0] = val
            frame[:, :, 1] = img1
            plt.imshow(frame)
            plt.show()

        x0, y0, z0, img0 = x1, y1, z1, img1

    # Save final image
    trace['img'] = img0

    return trace


def vis_trace(trace, image_size, roi_size, show_step=False):
    # Start point
    u = trace['u']
    s = trace['s']
    p0 = trace['s0']

    ds = [s1 - s0 for s0, s1 in zip(s, s[1:])]

    frame = np.zeros((image_size, image_size, 3))
    frame[int(p0[0]), int(p0[1]), 1] = p0[2]

    for control, state in zip(u, s):
        p1 = p0 + control * np.array([roi_size, roi_size, 1])

        # Last move
        rr, cc, val = line_aa(int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1]))
        frame[rr, cc, 0] = val

        # Current ellipse
        ellipse = ((state[0] * roi_size + p0[1], state[1] * roi_size + p0[0]),
                   (state[2] * roi_size, state[3] * roi_size),
                   (state[4] * 180))
        frame = cv2.ellipse(frame, ellipse, (0, 0, 1), 2)

        # Current point
        p0 = p1
        frame[int(p0[0]), int(p0[1]), 1] = p0[2]

        if show_step:
            plt.figure(1)
            plt.imshow(frame)
            # plt.show()

    frame[:, :, 1] = trace['img']
    plt.imshow(frame)
    plt.show()
    return frame


def trace_to_dxux(trace):
    """
    return: [dx, (s, u)]
    """
    u = trace['u']
    s = trace['s']

    dx = [x1 - x0 for x0, x1 in zip(s, s[1:])]
    x_w = [np.concatenate([uu, ss]) for uu, ss in zip(u, s)]
    samples = (np.array(dx), np.array(x_w[1:]))
    return samples


def get_supervised_wps_from_track(path, num_points):
    """
    从path中等间距提取num_points个点，作为监督点，包括起始点和终止点
    :param path:
    :param num_points:
    :return:
    """
    stride = int(np.ceil(path.shape[0] / (num_points - 1)))
    supervised_wps = path[::stride]
    while supervised_wps.shape[0] < num_points:  # use last point to complete the path
        supervised_wps = np.concatenate([supervised_wps, path[-1:]], axis=0)

    supervised_wps = np.concatenate([supervised_wps, np.zeros((supervised_wps.shape[0], 1))], axis=1)
    return supervised_wps


def cut_roi(target_image, position, roi_size):
    cur_pos = np.array([position[0], position[1]], dtype=np.int)
    img = np.pad(target_image, (int(roi_size / 2), int(roi_size / 2)), 'constant')
    cur_pos += np.array([roi_size / 2, roi_size / 2], dtype=np.int)
    roi = img[
          int(cur_pos[0] - roi_size / 2): int(cur_pos[0] + roi_size / 2),
          int(cur_pos[1] - roi_size / 2): int(cur_pos[1] + roi_size / 2)
          ]
    return roi


def refpath_to_actions(refpath, step_size, action_shape):
    delta = refpath[1:] - refpath[:-1]
    actions = np.round(delta / (0.5 * step_size))
    actions += action_shape // 2
    z = np.random.randint(low=0, high=5, size=(actions.shape[0], 1))
    actions = np.concatenate([actions, z], axis=-1)

    return actions


if __name__ == '__main__':
    agent = MypaintAgent({'brush_name': 'custom/slow_ink'})
    roi_size = 16 * 4
    image_size = 192 * 4
    cur_author = '1001-c'

    words = read_pot(cur_author)
    centralized_words = [centralize_char(word, dst_size=image_size, margin=5) for word in
                         words.values()]
    strokes = [s for w in centralized_words for s in w]
    strokes = [interpolate_stroke(s, roi_size // 2) for s in strokes]
    print(f'num of strokes:{len(strokes)}')

    trace = stroke_to_trace(strokes[130],
                            agent,
                            image_size,
                            roi_size,
                            display=False)
    vis_trace(trace, image_size, roi_size, True)

    samples = trace_to_dxux(trace)
    print(samples)
