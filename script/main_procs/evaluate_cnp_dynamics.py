import matplotlib.pyplot as plt
import numpy as np

from Data.data_process_lib import refpath_to_actions
from Env.main_env import MainEnv, Canvas
from Model.dynamics_model import CNPModel, ActionEmbedder
from Model.repr_model import ReprModel
from script.main_procs.hparams import define_hparams
from script.main_procs.sample_context import gen_context

if __name__ == '__main__':
    cfg = define_hparams()
    dynamics = CNPModel(cfg)
    dynamics.load_model()

    # Sample context points
    repr_model = ReprModel(cfg)
    embedder = ActionEmbedder(cfg)

    context_x, context_y = gen_context(cfg, repr_model, embedder, num_context_points=20)
    dynamics.set_repr(repr_model, embedder)
    dynamics.set_context(context_x, context_y)

    env = MainEnv(cfg)
    canvas = Canvas(cfg)
    canvas_gt = Canvas(cfg)

    for i in range(10):
        # -------------------
        canvas.reset()
        canvas_gt.reset()

        obs = env.reset()
        position = env.history[0]
        x, y = position[0], position[1]
        action_disp = cfg.action_shape // 2

        latent = repr_model.latent_encode(obs)
        ref_actions = refpath_to_actions(env.cur_ref_path,
                                         cfg.xy_grid,
                                         cfg.action_shape).tolist()

        # --------------------
        done = False
        while not done and len(ref_actions) > 0:
            action = ref_actions.pop(0)
            obs, rew, done, info = env.step(action)
            x += (action[0] - action_disp) * cfg.xy_grid / action_disp
            y += (action[1] - action_disp) * cfg.xy_grid / action_disp

            # Use latent to rollout
            embedding = embedder.transform([action])
            query_x = np.concatenate([latent, embedding], axis=-1)
            target_y = dynamics.predict(query_x)

            delta_latent = target_y['mu']
            latent += delta_latent
            delta = repr_model.latent_decode(latent).squeeze(axis=0).squeeze(axis=-1)
            canvas.place_delta(delta, (x, y))
            # canvas.render()

            latent_gt = repr_model.latent_encode(obs)
            delta_gt = repr_model.latent_decode(latent_gt).squeeze(axis=0).squeeze(axis=-1)
            canvas_gt.place_delta(delta_gt, (x, y))

        gt = env.mypaint_painter.get_img(shape=(cfg.image_size, cfg.image_size))
        plt.imshow(np.concatenate([canvas.frame, canvas_gt.frame, gt], axis=1))
        plt.show()

        # TODO: metrics on both Image and Latent to evaluate dynamics

