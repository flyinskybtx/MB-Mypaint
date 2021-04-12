from ray.rllib.evaluation import SampleBatchBuilder
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.offline.json_writer import JsonWriter
from tqdm import tqdm

from Data.data_process_lib import refpath_to_actions
from Data.Deprecated.core_config import *
from Data.Deprecated.windowed_env import WindowedCnnEnv

if __name__ == '__main__':
    env_config = {
        'image_size': experimental_config.image_size,
        'window_size': experimental_config.window_size,
        'xy_grid':experimental_config.xy_grid,
        'z_size': experimental_config.z_size,
        'brush_name': experimental_config.brush_name,
        'image_nums': experimental_config.image_nums,
        'action_shape': experimental_config.action_shape,
        'obs_size': experimental_config.obs_size,
    }
    env = WindowedCnnEnv(env_config)

    batch_builder = SampleBatchBuilder()
    writer = JsonWriter('../offline/windowed',
                        max_file_size=1024 * 1024 * 1024,
                        compress_columns=['obs', 'new_obs'])

    prep = get_preprocessor(env.observation_space)(env.observation_space)
    print("The preprocessor is", prep)

    eps_id = 0
    pbar = tqdm(total=1000)
    while eps_id < pbar.total:
        obs = env.reset()
        done = False
        t = 0

        reference_path = env.cur_ref_path
        actions = refpath_to_actions(reference_path,
                                     xy_size=experimental_config.obs_size,
                                     action_shape=experimental_config.action_shape)
        assert len(actions) > 0

        for action in actions:
            if done:
                break
            new_obs, rew, done, info = env.step(action)
            batch_builder.add_values(
                t=t,
                eps_id=eps_id,
                obs=prep.transform(obs),
                new_obs=prep.transform(new_obs),
                actions=action,
                dones=done,
                infos={},
            )
            obs = new_obs
            t += 1

        if batch_builder.count >= 5:
            writer.write(batch_builder.build_and_reset())
            eps_id += 1
            pbar.update(1)
