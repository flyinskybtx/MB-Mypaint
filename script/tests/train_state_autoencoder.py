import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from Data.Deprecated.core_config import experimental_config
from Data.Deprecated.representation_model import build_AE
from script.tests.train_image_autoencoder import OfflineDataGenerator


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
                frame = np.concatenate([y_pred, y_true], axis=1)  # concat along ys axis for view
                fig = plt.figure(f'Epoch:{epoch}')
                plt.imshow(np.round(frame))
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
