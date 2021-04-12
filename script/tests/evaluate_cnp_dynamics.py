import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from tqdm import tqdm

from Data.data_process_lib import refpath_to_actions
from Data.Deprecated.core_config import experimental_config
from Data.Deprecated.windowed_env import WindowedCnnEnv
from Data.Deprecated.cnn_model import LayerConfig
from Data.Deprecated.old_cnp_model import build_cnp_model
from script.tests.train_cnp_dynamics import CNPDataGenerator

if __name__ == '__main__':
    # --------- load models --------- #
    config = {
        'state_dims': 7,
        'action_dims': 3,
        'logits_dims': 8,
        'latent_encoder': {LayerConfig(fc=8, activation='relu')},
        'latent_decoder': {LayerConfig(fc=8, activation='relu')},
    }
    cnp_model = build_cnp_model(config)
    cnp_model.load_weights('../Model/checkpoints/cnp_model.h5')
    encoder = keras.models.load_model('../../Model/checkpoints/encoder')
    decoder = keras.models.load_model('../../Model/checkpoints/decoder')

    # ------ create rollout env -------- #
    env_config = {
        'image_size': experimental_config.image_size,
        'window_size': experimental_config.window_size,
        'z_size': experimental_config.z_size,
        'brush_name': experimental_config.brush_name,
        'image_nums': experimental_config.image_nums,
        'action_shape': experimental_config.action_shape,
    }
    env = WindowedCnnEnv(env_config)

    # ------- load context point ------- #
    generator_config = {
        'window_size': experimental_config.window_size,
        'batch_size': 1,
        'offline_data': '../Data/offline/windowed',
        'slots': ['obs', 'actions', 'new_obs'],
        'latent_encoder': '../Model/checkpoints/latent_encoder',
        'num_context': [5, 10],
    }
    train_data_generator = CNPDataGenerator(generator_config)
    context, _ = train_data_generator.__getitem__(0)
    context_x = context['context_x']
    context_y = context['context_y']

    # ---------- Evaluate for N num_episodes ------------- #
    for i in tqdm(range(10)):
        # ---------- get ref waypoints ------------ #
        obs = env.reset()  # obs = [cur, prev, tar, zs]
        cur_cnp, prev_cnp, tar, z = map(lambda x: np.squeeze(x, axis=-1), np.split(obs, 4, axis=-1))
        reference_path = env.cur_ref_path
        actions = refpath_to_actions(reference_path,
                                     xy_size=experimental_config.window_size,
                                     action_shape=experimental_config.action_shape)
        actions[:10, -1] = 4

        done = False
        t = 0

        for action in actions:
            if done:
                break
            # ---------- mlp rollout ---------- #
            cnp_obs = np.stack([cur_cnp, prev_cnp, tar, z], axis=-1)
            latent = encoder.predict(np.expand_dims(cnp_obs, axis=0))
            query_x = np.concatenate([latent, np.expand_dims(action, axis=0)], axis=-1)
            target_y = cnp_model.predict({
                'context_x': context_x,
                'context_y': context_y,
                'query_x': query_x,
            })
            delta_latent = target_y['mu']
            new_latent = latent + delta_latent
            img_pred = decoder.predict(new_latent)
            img_pred = np.squeeze(np.squeeze(img_pred, axis=(0, -1)))
            # img_pred = np.round(img_pred)  # convert image into BW

            cur_cnp, prev_cnp = img_pred, cur_cnp  # update cur & prev
            # ---------- environmental rollout ---------- #
            obs, rew, done, info = env.step(action)
            img_true, _, tar, z = map(lambda x: np.squeeze(x, axis=-1), np.split(obs, 4, axis=-1))  # update tar & zs
            t += 1

            # ---------- compare GT with CNP output ----------- #
            plt.imshow(np.concatenate([img_pred, img_true, tar], axis=-1))
            plt.title(f'{i}-({t})')
            plt.show()

            # TODO: place img into frames

            # TODO: horizon tests

            # TODO: connect model as one
