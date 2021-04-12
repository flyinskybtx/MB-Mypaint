import os
from glob import glob

import numpy as np
from ray.rllib.evaluation import SampleBatchBuilder
from ray.rllib.offline import JsonWriter
from tqdm import tqdm

from Data import DATA_DIR
from Data.HWDB.load_HWDB import get_waypoints_samples, HWDB_DIR
from Env.direct_env import DirectEnv
from script.main_procs.hparams import define_hparams

if __name__ == '__main__':
    env_config = define_hparams()
    physics = glob(os.path.join(DATA_DIR, f'offline/slow_ink/Physics'))[0]
    env_config.brush_info_file = os.path.join(physics, 'BrushInfo.myb')

    env = DirectEnv(env_config)

    strides = int(env_config.image_size / env_config.stride_size)
    margin = 0
    author = '1001-c.pot'
    strokes = get_waypoints_samples(os.path.join(HWDB_DIR, author),
                                    margin, strides, env_config.num_waypoints)
    env.reset()

    save_dir = os.path.join(DATA_DIR, 'HWDB/json', author)
    writer = JsonWriter(save_dir)
    batch_builder = SampleBatchBuilder()

    for stroke in tqdm(strokes):
        zs = np.random.randint(0, 10, size=(env_config.num_waypoints, 1))
        action = np.concatenate([stroke, zs], axis=-1).tolist()

        env.mypaint_painter.reset()
        obs, _, _, _ = env.step(action)

        # --- Save
        batch_builder.add_values(
            obs=obs,

            actions=action,
        )
        writer.write(batch_builder.build_and_reset())
