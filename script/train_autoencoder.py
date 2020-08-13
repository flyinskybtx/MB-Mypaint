from json import JSONDecodeError

import matplotlib.pyplot as plt
import numpy as np
from ray.rllib.models.model import restore_original_dimensions
from ray.rllib.offline import JsonReader
from tensorflow import keras

from Env.core_config import experimental_config
from Env.windowed_env import WindowedCnnEnv
from Model.representation_model import build_AE


class OfflineDataGenerator(keras.utils.Sequence):
    def __init__(self, reader, batch_size, obs_space):
        self.reader = reader
        self.batch_size = batch_size
        self.obs_space = obs_space

    def __getitem__(self, index):
        raise NotImplementedError

    def get_batch(self):
        source, action = [], []
        for i in range(self.batch_size):
            batch = self.reader.next()
            source.append(restore_original_dimensions(batch['obs'], self.obs_space))
            action.append(batch['actions'])
        source = np.concatenate(source)
        action = np.concatenate(action)
        return {'obs': source, 'actions': action}

    def __len__(self):
        return 100000


class AEDataGenerator(OfflineDataGenerator):
    def __getitem__(self, index):
        batch = self.get_batch()
        source = batch['obs']
        target = np.copy(source)[:, :, :, :1]  # Only select first channel for current image
        return source, target


class VisualiztionCallback(keras.callbacks.Callback):
    def __init__(self, data_generator):
        super().__init__()
        self.generator = data_generator

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % 5 != 0:
            return

        X, Y = self.generator.__getitem__(0)
        Y_ = self.model.predict(X)

        for y_pred, y_true in zip(Y_, Y):
            if np.max(y_true) > 0:
                frame = np.concatenate([y_pred, y_true], axis=1)  # concat along y axis for view
                fig = plt.figure(f'Epoch:{epoch}')
                plt.imshow(frame)
                plt.show()
                plt.close(fig)
                break


if __name__ == '__main__':
    model_saved_name = 'autoencoder'

    offline_dataset = '../Data/offline/windowed'
    reader = JsonReader(offline_dataset)

    env_config = {
        'image_size': experimental_config.image_size,
        'window_size': experimental_config.window_size,
        'z_size': experimental_config.z_size,
        'brush_name': experimental_config.brush_name,
        'image_nums': experimental_config.image_nums,
        'action_shape': experimental_config.action_shape,
    }

    windowed_env = WindowedCnnEnv(env_config)

    data_generator = AEDataGenerator(reader, batch_size=32, obs_space=windowed_env.observation_space)

    config = {
        'image_shape': (experimental_config.window_size, experimental_config.window_size, 4),
        'encoder_layers': [
            ('conv', [32, (2, 2), 1]),
            ('pool', [(2, 2), 2]),
            ('conv', [16, (2, 2), 2]),
            ('pool', [(2, 2), 2]),
            ('flatten', None),
            ('dense', 256),
        ],
        'decoder_layers': [
            ('dense', 256),
            ('dense', 1024),
            ('reshape', (8, 8, 16)),
            ('upsampling', (2, 2)),
            ('deconv', [32, (2, 2), 2]),
            ('upsampling', (2, 2)),
            ('deconv', [1, (2, 2), 1]),
        ],
        'bottleneck': experimental_config.bottleneck,
    }

    auto_encoder, encoder, decoder = build_AE(config)
    auto_encoder.compile(optimizer='adam', loss='mse')

    # Load previous trained model
    auto_encoder.load_weights(f'../Model/checkpoints/{model_saved_name}.h5')

    auto_encoder.fit_generator(
        data_generator, epochs=100, steps_per_epoch=100,
        validation_data=data_generator, validation_steps=100,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                f'../Model/checkpoints/{model_saved_name}.h5', save_best_only=True),
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=1, mode='auto', restore_best_weights=True),
            VisualiztionCallback(data_generator),
        ])

    encoder.save(f'../Model/checkpoints/{encoder.name}')
    decoder.save(f'../Model/checkpoints/{decoder.name}')
