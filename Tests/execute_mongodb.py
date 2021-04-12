import numpy as np
import json
import matplotlib.pyplot as plt

from Env.direct_env_v2 import RobotDirectEnvV2
from Main import _direct_env_config, _brush_config

env_config = _direct_env_config()
env_config.brush_config = _brush_config()

env = RobotDirectEnvV2(env_config)
img = env.reset()

with open('human_actions.json', 'r') as fp:
    human_actions = json.load(fp)

for i, points in human_actions.items():
    action = np.array(points)[:, :3]
    action = np.clip(action, 0, 1).tolist()

    img, _, _, _ = env.step(action)
    img = 1 - np.concatenate([img] * 3, axis=-1)
    img = np.rot90(img, )

    plt.imshow(img)
    plt.suptitle(i)
    # plt.show()
    plt.imsave(f"human_action_simulation_results/{i}.jpg", img)
