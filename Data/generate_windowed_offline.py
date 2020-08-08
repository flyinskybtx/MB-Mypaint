from ray.rllib.evaluation import SampleBatchBuilder
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.offline.json_writer import JsonWriter
from tqdm import tqdm

from Data.data_process import refpath_to_actions
from Env.core_config import *
from Env.windowed_env import WindowedCnnEnv

if __name__ == '__main__':
    env_config = {
        'image_size': experimental_config.image_size,
        'window_size': experimental_config.window_size,
        'z_size': experimental_config.z_size,
        'brush_name': experimental_config.brush_name,
        'image_nums': experimental_config.image_nums,
        'action_shape': experimental_config.action_shape,
    }
    env = WindowedCnnEnv(env_config)

    batch_builder = SampleBatchBuilder()
    writer = JsonWriter('./offline/windowed')

    prep = get_preprocessor(env.observation_space)(env.observation_space)
    print("The preprocessor is", prep)

    for eps_id in tqdm(range(1000)):
        obs = env.reset()
        prev_action = np.zeros_like(env.action_space.sample())
        prev_reward = 0
        done = False
        t = 0

        reference_path = env.cur_ref_path
        actions = refpath_to_actions(reference_path, experimental_config.window_size,
                                     action_shape=experimental_config.action_shape)

        for action in actions:
            if done:
                break
            new_obs, rew, done, info = env.step(action)
            batch_builder.add_values(
                t=t,
                eps_id=eps_id,
                agent_index=0,
                obs=prep.transform(obs),
                actions=action,
                action_prob=1.0,  # put the true action probability here
                rewards=rew,
                prev_actions=prev_action,
                prev_rewards=prev_reward,
                dones=done,
                infos={},
                new_obs=prep.transform(new_obs))
            obs = new_obs
            prev_action = action
            prev_reward = rew
            t += 1
        writer.write(batch_builder.build_and_reset())
