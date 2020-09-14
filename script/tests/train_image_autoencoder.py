import gym
import matplotlib.pyplot as plt
import numpy as np
from ray.rllib.models.model import restore_original_dimensions
from ray.rllib.offline import JsonReader
from tensorflow import keras

from Data.Deprecated.core_config import experimental_config
from Data.Deprecated.representation_model import build_AE


class OfflineDataGenerator(keras.utils.Sequence):
    def __init__(self, config: dict):
        self.reader = JsonReader(config['offline_data'])
        self.batch_size = config['batch_size']
        self.obs_space = gym.spaces.Box(low=0, high=1, dtype=np.float,
                                        shape=(config['obs_size'], config['obs_size'], 4))
        self.slots = config.setdefault('slots', [])

    def __getitem__(self, index):
        raise NotImplementedError

    def get_batch(self, batch_size):
        for i in range(batch_size):
            # initialize batch
            data = self.reader.next()
            if i == 0:
                batch = {k: [] for k in data.keys()}
            for k, v in data.items():
                if k in ['obs', 'new_obs', 'prev_obs']:
                    batch[k].append(restore_original_dimensions(v, self.obs_space))
                else:
                    batch[k].append(v)

        return {k: np.concatenate(v) for k, v in batch.items()}

    def __len__(self):
        return 100000


class AEDataGenerator(OfflineDataGenerator):
    def __getitem__(self, index):
        batch = self.get_batch(self.batch_size)
        source = batch['obs']
        target = np.copy(source)[:, :, :, :1]  # Only select first channel for current image
        return source, target


class VisualiztionCallback(keras.callbacks.Callback):
    def __init__(self, data_generator, frequency):
        super().__init__()
        self.generator = data_generator
        self.frequency = frequency

    def on_epoch_begin(self, epoch, logs=None, totol_count=10):
        if epoch % self.frequency != 0:
            return

        X, Y = self.generator.__getitem__(0)
        Y_ = self.model.predict(X)

        count = 0
        for y_pred, y_true in zip(Y_, Y):
            if np.max(y_true) > 0:
                frame = np.concatenate([y_pred, y_true], axis=1)  # concat along y axis for view
                fig = plt.figure(f'Epoch:{epoch}')
                plt.imshow(frame)
                plt.show()
                plt.close(fig)
                count += 1
            if count > totol_count:
                break


if __name__ == '__main__':
    model_saved_name = 'autoencoder'

    generator_config = {
        'window_size': experimental_config.window_size,
        'obs_size': experimental_config.obs_size,
        'batch_size': 16,
        'offline_data': '../Data/offline/windowed',
        'slots': ['obs', 'actions']
    }

    data_generator = AEDataGenerator(generator_config)

    config = {
        'image_shape': (experimental_config.obs_size, experimental_config.obs_size, 4),
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
            ('deconv', [32, (2, 2), 1]),
            ('deconv', [1, (2, 2), 1]),
        ],
        'bottleneck': experimental_config.bottleneck,
    }

    auto_encoder, encoder, decoder = build_AE(config)
    auto_encoder.compile(optimizer=keras.optimizers.Adam(5e-4), loss='mse')

    # Load previous trained model
    auto_encoder.load_weights(f'../Model/checkpoints/{model_saved_name}.h5')

    auto_encoder.fit_generator(
        data_generator, epochs=100, steps_per_epoch=100,
        validation_data=data_generator, validation_steps=100,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                f'../Model/checkpoints/{model_saved_name}.h5', save_best_only=True),
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, mode='auto', restore_best_weights=True),
            VisualiztionCallback(data_generator, frequency=3),
        ])

    encoder.save(f'../Model/checkpoints/{encoder.name}')
    decoder.save(f'../Model/checkpoints/{decoder.name}')
