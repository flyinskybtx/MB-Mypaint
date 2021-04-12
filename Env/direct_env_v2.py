import gym
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage
from skimage import transform

from Data.data_process_lib import load_all_stroke_pngs
from Env.agent import Agent
from Model import AttrDict
from utils.custom_rewards import rewards_dict


class RobotDirectEnvV2(gym.Env):
    def __init__(self, env_config: AttrDict):
        self.obs_size = env_config['obs_size']
        self.num_images = env_config['num_images']
        self.num_waypoints = env_config['num_waypoints']
        self.reward_names = env_config['rewards']
        self.split_view = env_config['split_view']

        # --- Spaces
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=(self.obs_size, self.obs_size, 1),
                                                dtype=np.float)
        self.action_space = gym.spaces.Box(low=0., high=1., shape=(3 * self.num_waypoints, ))

        # --- Load Images
        self.images = load_all_stroke_pngs()

        # --- Agent
        self.agent = Agent(env_config["brush_config"])

        # --- Reward_fn
        self.reward_fn = self.make_reward_fn(self.reward_names)

    def reset(self, **kwargs):
        """
        obs: cur, prev, tar, zs
        :return:
        """
        image_num = kwargs.setdefault('image_num', None)
        if image_num is None:
            image_num = np.random.randint(low=0, high=self.num_images - 1)

        # print(f'Image No.{image_num}\n')
        self.target_image = self.to_obs(self.images[image_num])
        return self.target_image

    def step(self, action: list):
        actions = np.array(action, dtype=np.float).reshape(-1, 3)
        self.actions = actions
        start_point = actions[0]
        start_point[-1] = 0
        actions = np.concatenate([np.expand_dims(start_point, axis=0), actions], axis=0)
        img = self.agent.execute(actions)
        self.obs = self.to_obs(img)

        rewards = self.reward_fn(self.target_image, self.obs)
        reward = 0 + np.sum(list(rewards.values()))

        return self.obs, reward, True, {}  # obs, accu_reward, done, info

    def render(self, **kwargs):
        wps_frame = np.zeros((self.obs_size, self.obs_size))
        wps = self.actions * np.array([self.obs_size - 1, self.obs_size - 1, 1])
        for wp in wps:
            wps_frame[int(wp[0]), int(wp[1])] = wp[2]
        kernel = np.ones((3, 3))
        wps_frame = scipy.ndimage.convolve(wps_frame, kernel, mode='constant', cval=0.0)
        wps_frame = np.expand_dims(wps_frame, axis=-1)

        if not self.split_view:
            plt.imshow(np.concatenate([wps_frame, self.target_image, self.obs], axis=-1))
            plt.show()
        else:
            plt.imshow(np.concatenate([wps_frame, self.target_image, self.obs], axis=1))
            plt.show()

    def to_obs(self, img):
        img = img.astype(np.float)
        img = transform.resize(img, (self.obs_size, self.obs_size))  # Resize to a smaller obs size
        img = img.reshape((self.obs_size, self.obs_size, 1))
        img = np.clip(img, 0, 1)
        return img

    @staticmethod
    def make_reward_fn(reward_names):
        def calculate_reward(target, obs):
            rewards = {}
            for name in reward_names:
                rewards[name] = rewards_dict[name](target, obs)
            return rewards

        return calculate_reward


def test_robot_directenv():
    config = AttrDict()

    config.brush_config = AttrDict()
    config.brush_config.brush_name = 'custom/slow_ink'
    config.brush_config.agent_name = 'Physics'

    config.direct_env_config = AttrDict()
    config.direct_env_config.num_xy = 24
    config.direct_env_config.num_z = 10
    config.direct_env_config.num_waypoints = 6
    config.direct_env_config.obs_size = 64
    config.direct_env_config.num_images = 64
    config.direct_env_config.rewards = [
        'img_cosine_reward',
        # 'img_mse_loss',
        # 'scale_loss',
        # 'curvature_loss',
        # 'iou_reward',
        # 'incremental_reward',
        # 'incremental_loss'
    ]
    config.direct_env_config.split_view = False
    config.direct_env_config.brush_config = config.brush_config

    env = RobotDirectEnv(config.direct_env_config)
    for _ in range(10):
        obs = env.reset()
        action = env.action_space.sample()
        print("Action:", action)
        env.step(action)
        env.render()


if __name__ == '__main__':
    test_robot_directenv()
