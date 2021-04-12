import os
from glob import glob
import tensorflow as tf

tf.executing_eagerly()

from Data import DATA_DIR, obs_to_delta
from Data.HWDB.load_HWDB import get_skeleton_paths_from_pot, HWDB_DIR
from Data.data_process_lib import skeleton_path_to_wps
from Env.canvas import SimpleCanvas
from Env.continuous_env import ContinuousEnv
from Model import MODEL_DIR
from Model.cnp_model import CNP
from Model.obs_encoder import ObsEncoder
from Model.obs_decoder import ObsDecoder
from script.main_procs.hparams import define_hparams
import numpy as np

from tune.continuous_way.data_utils import random_actions_from_skeleton, collect_data_from_actions

if __name__ == '__main__':
    cfg = define_hparams()

    obs_encoder = ObsEncoder(cfg, name=f'obs_encoder_{cfg.latent_size}')
    obs_encoder.build_graph(input_shape=(None, cfg.obs_size, cfg.obs_size, 1))
    obs_encoder.load_weights(os.path.join(MODEL_DIR, 'checkpoints', f'{obs_encoder.name}.h5'))
    obs_encoder.trainable = False

    obs_decoder = ObsDecoder(cfg, name=f'obs_decoder_{cfg.latent_size}')
    obs_decoder.build_graph(input_shape=(None, cfg.latent_size))
    obs_decoder.load_weights(os.path.join(MODEL_DIR, 'checkpoints', f'{obs_decoder.name}.h5'))
    obs_decoder.trainable = False

    cnp = CNP(cfg)
    cnp.build_graph(input_shape=[(None, None, cfg.latent_size + 3),
                                 (None, None, cfg.latent_size),
                                 (None, cfg.latent_size + 3)])
    cnp.summary()
    cnp.load_weights(filepath=os.path.join(MODEL_DIR, 'checkpoints/CNP/'))

    # ----------
    env_config = define_hparams()
    physics = glob(os.path.join(DATA_DIR, f'offline/slow_ink/Physics'))[0]
    env_config.brush_info_file = os.path.join(physics, 'BrushInfo.myb')

    env = ContinuousEnv(env_config)

    margin = 0
    author = '1001-c.pot'
    skeleton_paths = get_skeleton_paths_from_pot(os.path.join(HWDB_DIR, author),
                                                 margin, env_config.image_size, total_strokes=100)

    context_x = []
    context_y = []

    for i in range(10):
        sp = skeleton_paths[i]
        actions, start_point = random_actions_from_skeleton(sp, env_config.xy_grid, env_config.image_size)
        data = collect_data_from_actions(env, actions, start_point)
        for obs, new_obs, action in zip(data['obs'], data['new_obs'], data['actions']):
            x0 = obs_encoder.predict(obs_to_delta(np.expand_dims(obs, axis=0).astype(np.float32))).squeeze(axis=0)
            x1 = obs_encoder.predict(obs_to_delta(np.expand_dims(new_obs, axis=0).astype(np.float32))).squeeze(axis=0)
            u = action
            dx = x1 - x0
            xu = np.concatenate([x0, u], axis=-1)

            context_x.append(xu)
            context_y.append(dx)

    context_x = np.expand_dims(np.stack(context_x, axis=0), axis=0)
    context_y = np.expand_dims(np.stack(context_y, axis=0), axis=0)

    for i in range(11, 30):
        sp = skeleton_paths[i]
        actions, start_point = random_actions_from_skeleton(sp, env.xy_grid, env.image_size)
        data = collect_data_from_actions(env, actions, start_point)
        target_img = data['target_img']
        simple_canvas = SimpleCanvas(env.window_size, np.zeros_like(target_img), start_point=start_point)

        for obs, action, point in zip(data['obs'], data['actions'], data['points']):
            x0 = obs_encoder.predict(obs_to_delta(np.expand_dims(obs, axis=0).astype(np.float32))).squeeze(axis=0)
            u = action
            xu = np.concatenate([x0, u], axis=-1)
            query_x = np.expand_dims(xu, axis=0)
            dx = cnp.predict([context_x, context_y, query_x]).squeeze(axis=0)
            x1 = x0 + dx
            delta = obs_decoder.predict(np.expand_dims(x1, axis=0)).squeeze(axis=0).squeeze(axis=-1)

            frame = simple_canvas.attach(delta, point)

        import matplotlib.pyplot as plt

        plt.imshow(np.concatenate([simple_canvas.frame, target_img], axis=-1))
        plt.show()
