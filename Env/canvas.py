import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
import tensorflow as tf


class SimpleCanvas:
    def __init__(self, window_size, frame=None, start_point=None):
        self.frame = frame
        self.window_size = window_size

        if start_point is not None:
            self.x = start_point[0]
            self.y = start_point[1]

    def attach(self, delta, position):
        window = transform.resize(delta, (self.window_size, self.window_size))
        padded_frame = np.pad(self.frame, (self.window_size // 2, self.window_size // 2), 'constant')
        padded_frame[
            int(position[0]):int(position[0]) + self.window_size,
            int(position[1]):  int(position[1]) + self.window_size
        ] += window
        self.frame = padded_frame[
                         self.window_size // 2:-self.window_size // 2,
                         self.window_size // 2:-self.window_size // 2
                     ]
        return self.frame


class Canvas:
    def __init__(self, frame, delta, x, y, z, window_size, dynamics=None):
        self.frame = np.copy(frame).astype(np.float32)
        self.delta = np.copy(delta).astype(np.float32)
        self.x = x
        self.y = y
        self.z = z
        self.dynamics = dynamics
        self.half_window = window_size // 2

    def step(self, action):
        self._update_position(action)
        delta = self.dynamics.step(action)
        self._update_frame(delta)

    def _update_position(self, action):
        x, y, z = action
        self.x += x
        self.y += y
        self.z += z

    def _update_frame(self, delta):
        window = transform.resize(delta, (2 * self.half_window, 2 * self.half_window))

        padded_frame = np.pad(self.frame, (self.half_window, self.half_window), 'constant')
        padded_frame[self.x:self.x + 2 * self.half_window, self.y:self.y + 2 * self.half_window] += window
        self.frame = padded_frame[self.half_window:-self.half_window, self.half_window:-self.half_window]
        return self.frame

    def render(self):
        plt.imshow(self.frame)
        plt.show()


class CanvasFactory:
    def __init__(self, canvas, num_canvas, dynamics):
        self.window_size = 2 * canvas.half_window
        self.image_shape = canvas.frame.shape[0]
        self.frames = np.repeat((np.expand_dims(canvas.frame, axis=0)), repeats=num_canvas, axis=0)
        self.deltas = np.repeat((np.expand_dims(canvas.delta, axis=0)), repeats=num_canvas, axis=0)
        self.xs = canvas.x * np.ones((num_canvas, 1), dtype=np.float32)
        self.ys = canvas.y * np.ones((num_canvas, 1), dtype=np.float32)
        self.zs = canvas.z * np.ones((num_canvas, 1), dtype=np.float32)

        self.dynamics = dynamics

    def step(self, actions):
        self._update_position(actions)
        deltas = tf.squeeze(self.dynamics.step(actions, self.zs), axis=-1).numpy()
        self._update_frame(deltas)

    def _update_position(self, actions):
        self.xs += np.expand_dims(actions[:, 0], axis=-1)
        self.ys += np.expand_dims(actions[:, 1], axis=-1)
        self.zs += np.expand_dims(actions[:, 2], axis=-1)

        self.xs = np.clip(self.xs, 0, self.image_shape)
        self.ys = np.clip(self.ys, 0, self.image_shape)
        self.zs = np.clip(self.zs, 0, 1)

    def _update_frame(self, deltas):
        width, height = deltas.shape[1] // 2, deltas.shape[2] // 2
        shape = self.frames.shape
        padded_frame = np.zeros((shape[0], shape[1] + 2 * width, shape[2] + 2 * height))
        padded_frame[:, width:-width, height:-height] = self.frames

        for i, (d, x, y) in enumerate(zip(deltas, self.xs, self.ys)):
            x = int(x)
            y = int(y)
            padded_frame[i, x:x + 2 * width, y:y + 2 * height] += d

        self.frames = padded_frame[:, width:-width, height:-height]
        return self.frames
