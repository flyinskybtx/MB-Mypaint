import gym
import numpy as np
from ray.rllib.models.model import restore_original_dimensions
from ray.rllib.offline import JsonReader
from tensorflow import keras

from Env.core_config import experimental_config
from script.train_autoencoder import OfflineDataGenerator


class CNPDataGenerator(OfflineDataGenerator):
    def __init__(self, reader, batch_size, obs_space, encoder):
        super().__init__(reader, batch_size, obs_space)
        self.encoder = encoder

    def __getitem__(self, index):
        batch = self.get_batch()
        obs = batch['obs']  # batch size * M * M * 4 dim observation
        latent_0 = self.encoder.predict(obs)
        new_obs = batch['new_obs']  # batchsize * M * M * 4 dim observation
        latent_1 = self.encoder.predict(new_obs)
        actions = batch['actions']

        augmented_states = np.concatenate([latent_1, actions], axis=-1)
        delta_states = latent_1 - latent_0

        return augmented_states, delta_states

    def get_batch(self):
        obs, new_obs, actions = [], [], []
        for i in range(self.batch_size):
            batch = self.reader.next()
            obs.append(restore_original_dimensions(batch['obs'], self.obs_space))
            new_obs.append(restore_original_dimensions(batch['new_obs'], self.obs_space))
            actions.append(batch['actions'])
        obs = np.concatenate(obs)
        new_obs = np.concatenate(new_obs)
        actions = np.concatenate(actions)

        return {'obs': obs, 'new_obs': new_obs, 'actions': actions}


if __name__ == '__main__':
    encoder = keras.models.load_model('../Model/checkpoints/encoder')
    encoder.trainable = False
    offline_dataset = '../Data/offline/windowed'
    reader = JsonReader(offline_dataset)

    cnp_data_generator = CNPDataGenerator(
        reader,
        batch_size=32,
        obs_space=gym.spaces.Box(low=0,
                                 high=1,
                                 shape=(
                                     experimental_config.window_size,
                                     experimental_config.window_size,
                                     4),
                                 dtype=np.float),
        encoder=encoder
    )

    # TODO: build cnp model for latent states

    # TODO: convert training generator to context/target format for learning
