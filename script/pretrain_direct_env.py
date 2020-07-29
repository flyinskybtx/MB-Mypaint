import numpy as np

from Env.direct_env import DirectCnnEnv
from Model.cnn_model import CnnModel
from script.tune_direct_env import env_config, model_config

if __name__ == '__main__':
    direct_cnn_env = DirectCnnEnv(env_config)

    observation_space = direct_cnn_env.observation_space
    action_space = direct_cnn_env.action_space
    num_outputs = np.sum(action_space.nvec)
    print('Number of outputs: ', num_outputs)

    name = 'pretrain_custom_cnn'

    model = CnnModel(observation_space, action_space, num_outputs, model_config, name)
