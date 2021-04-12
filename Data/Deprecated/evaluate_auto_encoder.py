import random

import matplotlib.pyplot as plt
import numpy as np

from Data.Deprecated.auto_encoder_data import AEGenerator
from Data.data_process_lib import refpath_to_actions
from Env.main_env import MainEnv, Canvas
from Data.Deprecated.repr_model import ReprModel
from script.main_procs.hparams import define_hparams


def evaluate_repr(config, repr_model):
    env = MainEnv(config)

    canvas = Canvas(config)

    for i in range(10):
        canvas.reset()
        obs = env.reset()
        position = env.history[0]
        x, y = position[0], position[1]
        action_disp = config.action_shape // 2

        ref_actions = refpath_to_actions(env.cur_ref_path,
                                         config.xy_grid,
                                         config.action_shape).tolist()
        done = False
        while not done and len(ref_actions) > 0:
            action = ref_actions.pop(0)
            obs, rew, done, info = env.step(action)
            x += (action[0] - action_disp) * config.xy_grid / action_disp
            y += (action[1] - action_disp) * config.xy_grid / action_disp
            mu, sigma = repr_model.latent_encode(obs)
            delta = repr_model.latent_decode(mu).squeeze(axis=0).squeeze(axis=-1)

            canvas.place_delta(delta, (x, y))
            # canvas.render()
        gt = env.mypaint_painter.get_img(shape=(config.image_size, config.image_size))
        plt.imshow(np.concatenate([canvas.frame, gt], axis=1))
        plt.show()


def evaluate_continuity(repr_model):
    train_data = AEGenerator(savedir='offline/random', batch_size=64, is_encoder=True, max_samples=1e5)
    samples = []
    for j in range(10):
        if len(samples) > 2:
            break
        delta_cur, _ = train_data.__getitem__(j)
        samples += [ss for ss in delta_cur if np.sum(ss) > 0]

    for _ in range(5):
        frame = []
        idx = list(range(len(samples)))
        random.shuffle(idx)
        sample1, sample2 = samples[idx[0]], samples[idx[1]]

        mu1, sigma1 = repr_model.latent_encoder.predict(np.expand_dims(sample1, axis=0))
        mu2, sigma2 = repr_model.latent_encoder.predict(np.expand_dims(sample2, axis=0))

        num_interps = 10
        plt.imshow(sample1.squeeze(axis=-1))
        plt.suptitle('sample_1')
        plt.show()
        for i in range(0, num_interps + 1):
            mu = mu1 + (mu2 - mu1) * i / num_interps
            delta = repr_model.latent_decode(mu)
            frame.append(delta.squeeze(axis=-1).squeeze(axis=0))
            # plt.imshow(deltas.squeeze(axis=-1).squeeze(axis=0))
            # plt.suptitle(f'sample_{i}/{num_interps}')
            # plt.show()

        plt.imshow(sample2.squeeze(axis=-1))
        plt.suptitle('sample_2')
        plt.show()

        frame = np.concatenate(frame, axis=-1)
        plt.imshow(frame)
        plt.show()


if __name__ == '__main__':
    cfg = define_hparams()
    cfg.train_latent_encoder = False
    cfg.train_decoder = False
    repr_model = ReprModel(cfg)

    evaluate_repr(cfg, repr_model)
    # evaluate_continuity(repr_model)
