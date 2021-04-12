import itertools
import os
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import line
from tqdm import tqdm

HWDB_DIR = Path(__file__).parent / 'pot'
DST_SIZE = 192


def translate_stroke(content):
    """ Unpack a stroke encoding to a list of (X,Y) waypoints """

    points = [content[p:p + 4] for p in range(0, len(content), 4)]
    # Switch xs and ys
    return np.array([[struct.unpack('h', p[2:])[0], struct.unpack('h', p[:2])[0]] for p in points])


def read_pot(filename):
    """Read *.pot file, return {word:stroke}"""

    author = filename.strip('.pot').split('/')[-1]
    with open(filename, 'rb') as fp:
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
    char_strokes = [np.clip(s, 0, dst_size - 1) for s in char_strokes]
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


def test():
    strokes = read_pot(os.path.join(HWDB_DIR, '1001-c.pot'))
    margin = 0
    dst_size = 32
    char_ss = strokes['没']
    char_ss = centralize_char(char_ss, dst_size, margin)
    vis_char(char_ss, dst_size)


def get_waypoints_samples(filename, margin, dst_size, number=6):
    all_strokes = []

    chars = read_pot(filename)
    for key in chars.keys():
        char_strokes = centralize_char(chars[key], dst_size, margin)
        for stroke in char_strokes:
            if stroke.shape[0] > number:
                index = np.arange(stroke.shape[0])
                index = sorted(np.random.choice(index, 6, replace=False))
                stroke = stroke[index]
                all_strokes.append(stroke)

            elif stroke.shape[0] < number:
                stroke = interpolate_stroke(stroke, interval=1)
                if stroke.shape[0] < number:
                    continue
                else:
                    index = np.arange(stroke.shape[0])
                    index = sorted(np.random.choice(index, number, replace=False))
                    stroke = stroke[index]
                    all_strokes.append(stroke)
    return all_strokes


def get_skeleton_paths_from_pot(filename, margin, dst_size, total_strokes=10000):
    """

    Args:
        filename:
        margin:
        dst_size:
        total_strokes:

    Returns:
        skeleton_paths: list
    """
    skeleton_paths = []
    chars = read_pot(filename)
    all_strokes = []
    for key in tqdm(chars.keys(), desc="Load characters: "):
        if len(all_strokes)>total_strokes:
            break

        char_strokes = centralize_char(chars[key], dst_size, margin)
        for stroke in char_strokes:
            stroke = stroke.astype(np.float)
            if len(stroke) >= 2:  # 排除掉单个点情况
                all_strokes.append(stroke)
    if len(all_strokes) > total_strokes:
        all_strokes = all_strokes[:total_strokes]

    for stroke in tqdm(all_strokes, desc="Make waypoints: "):
        new_stroke = []
        for p1, p2 in zip(stroke, stroke[1:]):
            num_p = int(max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1])))
            _path = [np.round(p1 + i * (p2 - p1) / num_p, decimals=0).astype(int) for i in range(num_p + 1)]
            new_stroke.append(_path)

        skeleton_paths.append(new_stroke)
    return skeleton_paths


if __name__ == '__main__':
    # test()
    all_strokes = get_waypoints_samples(os.path.join(HWDB_DIR, '1001-c.pot'), 0, 32)
