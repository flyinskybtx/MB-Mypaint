import numpy as np

from Data.data_process_lib import refpath_to_actions
from Env.main_env import MainEnv
from Model.dynamics_model import ActionEmbedder
from Model.repr_model import ReprModel
from script.main_procs.hparams import define_hparams


def gen_context(env_config, repr_model: ReprModel, embed: ActionEmbedder, num_context_points=15):
    """

    :param env:
    :param num_context_points:
    :return: context_x, context_y
    """
    # ------------------- Generate Context ------------------------- #
    # Get context points
    env = MainEnv(env_config)

    context_x = []
    context_y = []

    while len(context_x) < num_context_points:
        obs = env.reset()
        latent = repr_model.latent_encode(obs)
        ref_actions = refpath_to_actions(env.cur_ref_path,
                                         env_config.xy_grid,
                                         env_config.action_shape).tolist()
        done = False
        while not done and len(ref_actions) > 0:
            action = ref_actions.pop(0)
            new_obs, rew, done, info = env.step(action)
            new_latent = repr_model.latent_encode(new_obs)
            delta = new_latent - latent
            embedding = embed.transform([action])
            context_x.append(np.concatenate([latent, embedding], axis=-1))
            context_y.append(delta)
            latent = np.copy(new_latent)

    idx = np.arange(len(context_x))
    np.random.seed(0)
    np.random.shuffle(idx)
    context_x = np.concatenate(context_x, axis=0)[idx[:num_context_points]]  # num_points * (state_dim + action_dim)
    context_y = np.concatenate(context_y, axis=0)[idx[:num_context_points]]  # num_points * state_dim
    return context_x, context_y


if __name__ == '__main__':
    cfg = define_hparams()
    repr_model = ReprModel(cfg)
    action_embed = ActionEmbedder(cfg)

    context_x, context_y = gen_context(cfg, repr_model, action_embed, num_context_points=20)
