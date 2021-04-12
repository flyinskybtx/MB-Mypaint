import os
import random
import sys
from glob import glob

import numpy
from ray.rllib.evaluation import SampleBatchBuilder
from ray.rllib.offline import JsonWriter
from tqdm import trange

from Data.data_process_lib import refpath_to_actions
from Env.main_env import MainEnv
from script.main_procs.hparams import define_hparams
from utils import BRUSH_DIR

try:
    from lib import mypaintlib, tiledsurface
    from lib.config import mypaint_brushdir
    import lib.brush as mypaint_brush
except ModuleNotFoundError:
    sys.path.append('/home/flyinsky/.local/lib/mypaint')
    from lib import mypaintlib, tiledsurface
    from lib.config import mypaint_brushdir
    import lib.brush as mypaint_brush

from Data import DATA_DIR

CATEGORIES = ['radius_logarithmic', 'slow_tracking_per_dab']


def create_physics(brush_name='custom/slow_ink'):
    base_dir = os.path.join(DATA_DIR, '../../Data/offline', brush_name.split('/')[-1])
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'Physics'), exist_ok=True)
    src_file = os.path.join(BRUSH_DIR, f'{brush_name}.myb')
    tar_file = os.path.join(base_dir, 'Physics/BrushInfo.myb')
    os.system(f"cp {src_file} {tar_file}")

    print(f'Created physics: {brush_name.split("/")[-1]}  in DIR: {base_dir}.')


def create_simulators(physics_name, num_sims=20):
    base_dir = os.path.join(DATA_DIR, '../../Data/offline', physics_name)
    brush_info = mypaint_brush.BrushInfo()

    with open(os.path.join(base_dir, 'Physics/BrushInfo.myb'), 'r') as fp:
        brush_info.from_json(fp.read())

    for i in range(num_sims):
        for category in CATEGORIES:
            points = brush_info.get_points(category, 'pressure')
            points[1][1] *= float(random.uniform(0.8, 1.2))
            points[2][1] *= float(random.uniform(0.8, 1.2))
            # points[:, 1] *= numpy.random.uniform(0.8, 1.2, size=(3,))
            # points = numpy.round(points, decimals=2).tolist()
            brush_info.set_points(category, 'pressure', points)
        os.makedirs(os.path.join(base_dir, f'Sim{i}'), exist_ok=True)
        with open(os.path.join(base_dir, f'Sim{i}/BrushInfo.myb'), 'w') as fp:
            fp.write(brush_info.to_json())

    print(f'Created {num_sims} simulators.')


def collect_samples(config, episodes=500, policy=None, clean_exist=False):
    env = MainEnv(config)
    batch_builder = SampleBatchBuilder()
    save_dir = os.path.dirname(config.brush_info_file)
    writer = JsonWriter(save_dir)
    rewards = []

    for eps_id in trange(episodes):
        eps_rew = 0
        obs = env.reset()
        done = False
        ref_actions = refpath_to_actions(env.cur_ref_path,
                                         config.xy_grid,
                                         config.action_shape)
        for action in ref_actions:
            assert numpy.min(action) >= 0 and numpy.max(action) < config.action_shape
            if done:
                break
            new_obs, rew, done, info = env.step(action)
            # if numpy.max(new_obs[:, :, 0]) > 0 or numpy.max(obs[:, :, 0]) > 0:  # If not empty image
            batch_builder.add_values(
                obs=obs,
                new_obs=new_obs,
                actions=action,
            )
            obs = new_obs.copy()
            eps_rew += rew
        rewards.append(eps_rew)

        if batch_builder.buffers:  # Buffer is not empty
            writer.write(batch_builder.build_and_reset())

    return rewards


if __name__ == '__main__':
    brush_name = 'custom/slow_ink'
    create_physics(brush_name=brush_name)
    create_simulators(physics_name=brush_name.split('/')[-1], num_sims=20)

    cfg = define_hparams()
    simulators = sorted(glob(os.path.join(DATA_DIR, f'../../Data/offline/slow_ink', '*')))
    for sim in simulators:
        cfg.brush_info_file = os.path.join(sim, 'BrushInfo.myb')
        rewards = collect_samples(cfg)
        print(f"Finish {sim.split('/')[-1]}, reward min = {numpy.mean(rewards)}")
