import os
from glob import glob

from Controllers.dynamics import Dynamics
from Data import DATA_DIR
from Data.data_process_lib import refpath_to_actions
from Env.main_env import MainEnv
from Model import MODEL_DIR
from Model.action_embedder import ActionEmbedder
from Model.mlp_model import MLP
from Model.obs_encoder import ObsEncoder
from Model.obs_decoder import ObsDecoder
from Planner.cem_planner import CemPlanner
from script.main_procs.hparams import define_hparams

if __name__ == '__main__':
    cfg = define_hparams()

    encoder = ObsEncoder(cfg, name=f'obs_encoder_{cfg.latent_size}')
    encoder.build_graph(input_shape=(None, cfg.obs_size, cfg.obs_size, 1))
    encoder.load_weights(os.path.join(MODEL_DIR, 'checkpoints', f'{encoder.name}.h5'))

    decoder = ObsDecoder(cfg, name=f'obs_decoder_{cfg.latent_size}')
    decoder.build_graph(input_shape=(None, cfg.latent_size))
    decoder.load_weights(os.path.join(MODEL_DIR, 'checkpoints', f'{decoder.name}.h5'))

    embedder = ActionEmbedder(cfg)
    embedder.build(input_shape=(None, 3))

    predictor = MLP(cfg)
    predictor.build_graph(input_shape=[(None, cfg.latent_size + 15), (None, 1)])
    predictor.load_weights(os.path.join(MODEL_DIR, 'checkpoints', f'{predictor.name}.h5'))

    dynamics = Dynamics(encoder, decoder, embedder, predictor)

    physical_system = glob(os.path.join(DATA_DIR, f'offline/slow_ink/Physics'))[0]
    cfg.brush_info_file = os.path.join(physical_system, 'BrushInfo.myb')
    env = MainEnv(cfg)
    obs = env.reset(image_num=3)

    action_space = env.action_space
    reward_fn = env.reward_fn
    planner = CemPlanner(action_space, reward_fn, dynamics, horizon=10, num_particles=50)

    done = False

    ref_path = env.ref_path
    ref_actions = refpath_to_actions(ref_path,
                                     cfg.xy_grid,
                                     cfg.action_shape)
    for i in range(5):
        x, y, _ = ref_actions[i]
        z = 3
        env.step([x, y, z])
    env.render()

    while not done:
        canvas = env.to_canvas()
        planner.update_canvas(canvas)

        action, reward = planner.plan()
        print(f'Current action: {action}, Reward: {reward}')

        obs, reward, done, info = env.step(action)
        env.render()
