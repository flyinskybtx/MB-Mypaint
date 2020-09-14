from ray.rllib.evaluation import SampleBatchBuilder
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.offline.json_writer import JsonWriter
from tqdm import tqdm
import os

from Data.data_process_lib import extract_skeleton_trace, get_supervised_wps_from_track
from Data.Deprecated.core_config import experimental_config
from Env.direct_env import DirectCnnEnv

env_config = {
    'image_size': experimental_config.image_size,
    'stride_size': experimental_config.stride_size,
    'stride_amplify': experimental_config.stride_amplify,
    'z_size': experimental_config.z_size,
    'brush_name': experimental_config.brush_name,
    'num_keypoints': experimental_config.num_keypoints,
    'image_nums': experimental_config.image_nums,
}

if __name__ == '__main__':
    batch_builder = SampleBatchBuilder()

    save_dir = '../offline/direct'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    writer = JsonWriter(save_dir)

    env = DirectCnnEnv(env_config)
    prep = get_preprocessor(env.observation_space)(env.observation_space)

    for eps_id in tqdm(range(1000)):
        obs = env.reset()
        t = 0

        reference_path = extract_skeleton_trace(env.target_image, experimental_config.stride_size, discrete=True)
        action = get_supervised_wps_from_track(reference_path, experimental_config.num_keypoints).reshape(-1)

        new_obs, rew, done, info = env.step(action.reshape(-1))

        batch_builder.add_values(
            t=t,
            eps_id=eps_id,
            agent_index=0,
            obs=prep.transform(obs),
            actions=action,
            action_prob=1.0,  # put the true action probability here
            rewards=rew,
            dones=done,
            infos={},
            new_obs=prep.transform(new_obs))

        obs = new_obs
        t += 1

        writer.write(batch_builder.build_and_reset())
