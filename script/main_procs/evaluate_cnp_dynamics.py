from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import numpy as np

from Data.data_process_lib import refpath_to_actions
from Env.main_env import MainEnv, Canvas
from Model.dynamics_model import CNPModel
from Model.action_embedder import ActionEmbedder
from Model.repr_model import ReprModel
from script.main_procs.hparams import define_hparams
from script.main_procs.sample_context import gen_context


def evaluate_dynamics(cfg, dynamics, repr_model, embedder, num_context=15, hijack=None):
    context_x, context_y = gen_context(cfg, repr_model, embedder, num_context_points=num_context)
    dynamics.set_repr(repr_model, embedder)
    dynamics.set_context(context_x, context_y)

    env = MainEnv(cfg)
    canvas = Canvas(cfg)
    canvas_gt = Canvas(cfg)

    iter = 0
    while iter < 10:
        # -------------------
        canvas.reset()
        canvas_gt.reset()

        obs = env.reset(image_num=6)
        position = env.history[0]
        x, y = position[0], position[1]
        action_disp = cfg.action_shape // 2

        ref_actions = refpath_to_actions(env.cur_ref_path,
                                         cfg.xy_grid,
                                         cfg.action_shape).tolist()
        done = False
        latent = None
        # --------------------
        if hijack is None:
            latent = repr_model.latent_encode(obs)

        for i, action in enumerate(ref_actions):
            if done:
                break
            obs, rew, done, info = env.step(action)
            x += (action[0] - action_disp) * cfg.xy_grid / action_disp
            y += (action[1] - action_disp) * cfg.xy_grid / action_disp

            # AE rollout
            latent_gt = repr_model.latent_encode(obs)
            delta_gt = repr_model.latent_decode(latent_gt).squeeze(axis=0).squeeze(axis=-1)
            canvas_gt.place_delta(delta_gt, (x, y))
            if hijack is not None and i < hijack:
                canvas.place_delta(delta_gt, (x, y))

            if i == hijack:
                latent = np.copy(latent_gt)

            #  Dynamics rollout
            if i >= hijack:
                embedding = embedder.transform([action])
                query_x = np.concatenate([latent, embedding], axis=-1)
                target_y = dynamics.predict(query_x)

                delta_latent = target_y['mu']
                latent += delta_latent
                delta = repr_model.latent_decode(latent).squeeze(axis=0).squeeze(axis=-1)
                canvas.place_delta(delta, (x, y))
                # canvas.render()

        gt = env.mypaint_painter.get_img(shape=(cfg.image_size, cfg.image_size))
        if np.any(gt > 0):
            iter += 1

            plt.figure()
            plt.suptitle('cnp/ae/gt')
            plt.subplot(211)
            plt.imshow(np.concatenate([canvas.frame, canvas_gt.frame, gt], axis=1))
            plt.subplot(212)
            plt.imshow(np.concatenate([canvas.frame, canvas_gt.frame, gt], axis=1) > 0)
            plt.show()


if __name__ == '__main__':
    cfg = define_hparams()
    dynamics = CNPModel(cfg)
    dynamics.load_model()

    # Sample context points
    repr_model = ReprModel(cfg)
    embedder = ActionEmbedder(cfg)

    # TODO: metrics on both Image and Latent to evaluate dynamics
    evaluate_dynamics(cfg, dynamics, repr_model, embedder, num_context=100, hijack=5)
