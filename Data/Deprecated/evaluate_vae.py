import os
import os.path as osp
import random
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

from Data import DATA_DIR, obs_to_delta
from Data.Deprecated.auto_encoder_data import AEGenerator
from Data.data_process_lib import refpath_to_actions
from Data.vae_data import get_all_vae_samples
from Env.main_env import MainEnv, Canvas
from Model import MODEL_DIR
from Model.vae_model import VAE
from Model.obs_encoder import ObsEncoder
from Model.obs_decoder import ObsDecoder
from script.main_procs.hparams import define_hparams


def evaluate_vae(config, encoder, decoder):
    env = MainEnv(config)
    canvas = Canvas(config)

    for i in range(3):
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
            mu, sigma = encoder.predict(np.expand_dims(obs, axis=0))
            delta = decoder.predict(mu).squeeze(axis=0).squeeze(axis=-1)

            canvas.place_delta(delta, (x, y))
            # canvas.render()
        gt = env.mypaint_painter.get_img(shape=(config.image_size, config.image_size))
        if np.max(gt) > 0 or np.max(canvas.frame) > 0:
            plt.imshow(np.concatenate([canvas.frame, gt], axis=1))
            plt.show()


def evaluate_continuity(encoder, decoder):
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

        mu1, sigma1 = encoder(np.expand_dims(sample1, axis=0))
        mu2, sigma2 = encoder(np.expand_dims(sample2, axis=0))

        num_interps = 10
        plt.imshow(sample1.squeeze(axis=-1))
        plt.suptitle('sample_1')
        plt.show()
        for i in range(0, num_interps + 1):
            mu = mu1 + (mu2 - mu1) * i / num_interps
            delta = decoder.predict(mu)
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


def evaluate_sigma(config, encoder):
    env = MainEnv(config)
    canvas = Canvas(config)

    for i in range(3):
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
            mu, log_var = encoder.predict(np.expand_dims(obs, axis=0))
            delta = decoder.predict(mu).squeeze(axis=0).squeeze(axis=-1)
            sigma = np.exp(0.5 * log_var)
            print(f"Mu: {mu} - Max: {mu.max()}, Min: {mu.min()}")
            print(f"Sigma: {sigma} - Max: {sigma.max()}, Min: {sigma.min()}")


if __name__ == '__main__':
    cfg = define_hparams()
    cfg.is_vae=True

    encoder = ObsEncoder(cfg, name=f'obs_encoder_{cfg.latent_size}')
    decoder = ObsDecoder(cfg, name=f'obs_decoder_{cfg.latent_size}')
    vae = VAE(encoder, decoder)

    encoder.build_graph(input_shape=(None, cfg.obs_size, cfg.obs_size, 4))
    encoder.summary()
    decoder.build_graph(input_shape=(None, cfg.latent_size))
    decoder.summary()
    encoder.load_weights(osp.join(MODEL_DIR, f'checkpoints/{encoder.name}.h5'))
    decoder.load_weights(osp.join(MODEL_DIR, f'checkpoints/{decoder.name}.h5'))

    phys = glob(os.path.join(DATA_DIR, f'offline/slow_ink', 'Phy*'))
    vis_data = get_all_vae_samples(phys, train_decoder=False)['obs']
    vis_data = obs_to_delta(vis_data)

    evaluate_vae(cfg, encoder, decoder)
    evaluate_continuity(encoder, decoder)
    evaluate_sigma(cfg, encoder)
