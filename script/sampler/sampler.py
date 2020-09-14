import struct
from random import random
import numpy as np

from utils.mypaint_agent import MypaintPainter


def make_stroke(content):
    points = [content[p:p + 4] for p in range(0, len(content), 4)]
    # Switch x and y
    return np.array([[struct.unpack('h', p[2:])[0], struct.unpack('h', p[:2])[0]] for p in points])


def make_character(content, author):
    sample_size = struct.unpack('H', content[:2])[0]
    assert sample_size == len(content) + 4, f'{sample_size}, {len(content)}'

    word = content[2:6]
    word = word[1::-1].decode('GBK')
    stroke_num = struct.unpack('H', content[6:8])[0]

    strokes = content[8:]
    strokes = strokes.split(b'\xff\xff\x00\x00')[:-1]
    strokes = [make_stroke(s) for s in strokes]

    assert len(strokes) == stroke_num
    return Character(word, strokes, author)


def load_from_pot(filename):
    author = filename.strip('.pot').split('/')[-1]
    with open(filename, 'rb') as fp:
        content = fp.read()
        content = content.split(b'\xff\xff\xff\xff')[:-1]
        return [make_character(c, author) for c in content]


def get_stroke():
    pass


if __name__ == '__main__':
    agent = MypaintPainter({'brush_name': 'custom/slow_ink'})
    stroke = get_stroke()

    # Initialize
    agent.reset()
    x0, y0 = stroke[0]
    z0 = 0
    agent.paint(x0, y0, z0)
    prev_img = agent.get_img()
    samples = []
    for p0, p in stroke:
        # process
        x, y = p[:2]
        z = random.random()
        agent.paint(x, y, z)
        img = agent.get_img()
        samples.append((x - x0, y - y0, z - z0, img - prev_img))

        # update
        prev_img, x0, y0, z0 = img, x, y, z
