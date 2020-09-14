import matplotlib.pyplot as plt
import numpy as np

from Data.data_process_lib import refpath_to_actions
from Env.main_env import MainEnv, Canvas
from Model.repr_model import ReprModel
from script.main_procs.hparams import define_hparams

if __name__ == '__main__':
    cfg = define_hparams()
    cfg.train_latent_encoder = False
    cfg.train_decoder = False
    repr_model = ReprModel(cfg)

    env = MainEnv(cfg)

    canvas = Canvas(cfg)

    for i in range(10):
        canvas.reset()
        obs = env.reset()
        position = env.history[0]
        x, y = position[0], position[1]
        action_disp = cfg.action_shape // 2

        latent = repr_model.latent_encode(obs)
        ref_actions = refpath_to_actions(env.cur_ref_path,
                                         cfg.xy_grid,
                                         cfg.action_shape).tolist()
        done = False
        while not done and len(ref_actions) > 0:
            action = ref_actions.pop(0)
            obs, rew, done, info = env.step(action)
            x += (action[0] - action_disp) * cfg.xy_grid / action_disp
            y += (action[1] - action_disp) * cfg.xy_grid / action_disp
            latent = repr_model.latent_encode(obs)
            delta = repr_model.latent_decode(latent).squeeze(axis=0).squeeze(axis=-1)

            canvas.place_delta(delta, (x, y))
            # canvas.render()
        gt = env.mypaint_painter.get_img(shape=(cfg.image_size, cfg.image_size))
        plt.imshow(np.concatenate([canvas.frame, gt], axis=1))
        plt.show()
