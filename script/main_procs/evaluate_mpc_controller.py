import matplotlib.pyplot as plt
import numpy as np

from Env.main_env import MainEnv
from Model.action_embedder import ActionEmbedder
from Model.dynamics_model import CNPModel
from Model.mpc_model import make_reward_fn, ShootingMPCController
from Model.repr_model import ReprModel
from script.main_procs.hparams import define_hparams
from script.main_procs.sample_context import gen_context

if __name__ == '__main__':
    cfg = define_hparams()
    repr_model = ReprModel(cfg)
    dynamics = CNPModel(cfg)
    embedder = ActionEmbedder(cfg)
    dynamics.load_model()
    context_x, context_y = gen_context(cfg, repr_model, embedder, num_context_points=100)
    dynamics.set_repr(repr_model, embedder)
    dynamics.set_context(context_x, context_y)

    env = MainEnv(cfg)
    action_disp = cfg.action_shape // 2
    obs = env.reset(image_num=3)
    reward_fn = make_reward_fn(env.target_image)

    mpc_controller = ShootingMPCController(cfg, env.action_space, dynamics, repr_model, embedder, reward_fn,
                                           horizon=10, samples=500)
    # mpc_controller = CemMPCController(cfg, env.action_space, dynamics, repr_model, embedder,
    #                                   reward_fn, horizon=10, samples=50)
    for i in range(100):
        mpc_controller.preset(env.canvas, env.xy)
        action = mpc_controller.next_action(obs, print_expectation=True)
        obs, rew, done, info = env.step(action)

        plt.imshow(np.concatenate([env.canvas.frame, env.target_image], axis=1))
        plt.suptitle(f'Step-{i}')
        plt.show()
        if done:
            break
