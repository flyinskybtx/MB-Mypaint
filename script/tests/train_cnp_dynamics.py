import numpy as np
from tensorflow import keras

from Data.Deprecated.core_config import experimental_config
from Data.Deprecated.cnn_model import LayerConfig
from Data.Deprecated.old_cnp_model import build_cnp_model, dist_logp, stats, dist_mse
from script.tests.train_image_autoencoder import OfflineDataGenerator


class CNPDataGenerator(OfflineDataGenerator):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = keras.models.load_model(config['latent_encoder'])
        self.encoder.trainable = False
        self.min_context, self.max_context = config['num_context']
        self.num_context = np.random.randint(self.min_context, self.max_context)

    def __getitem__(self, index):
        batch = self.get_batch(self.num_context + self.batch_size)
        obs = batch['obs']  # batch size * M * M * 4 dim observation
        new_obs = batch['new_obs']  # batchsize * M * M * 4 dim observation
        actions = batch['actions']  # batchsize * 3

        latent_0 = self.encoder.predict(obs)
        latent_1 = self.encoder.predict(new_obs)
        delta_states = latent_1 - latent_0  # Y
        augmented_states = np.concatenate([latent_1, actions], axis=-1)  # X

        idx = np.arange(obs.shape[0])
        np.random.shuffle(idx)

        context_x = np.repeat(np.expand_dims(augmented_states[idx[:self.num_context]], axis=0),
                              self.batch_size, axis=0)
        context_y = np.repeat(np.expand_dims(delta_states[idx[:self.num_context]], axis=0),
                              self.batch_size, axis=0)
        query_x = augmented_states[idx[-self.batch_size:]]
        target_y = delta_states[idx[-self.batch_size:]]

        return {'context_x': context_x, 'context_y': context_y, 'query_x': query_x}, target_y

    def on_epoch_end(self):
        self.num_context = np.random.randint(self.min_context, self.max_context)


def test_data_generator():
    generator_config = {
        'window_size': experimental_config.window_size,
        'batch_size': 16,
        'offline_data': '../Data/offline/windowed',
        'slots': ['obs', 'actions', 'new_obs'],
        'latent_encoder': '../Model/checkpoints/latent_encoder',
        'num_context': [5, 10],
    }
    cnp_data_generator = CNPDataGenerator(generator_config)

    batch = cnp_data_generator.__getitem__(0)
    print(batch[0]['context_x'].shape, batch[0]['context_y'].shape, batch[0]['query_x'].shape, batch[1].shape)


if __name__ == '__main__':
    generator_config = {
        'window_size': experimental_config.window_size,
        'batch_size': 16,
        'offline_data': '../Data/offline/windowed',
        'slots': ['obs', 'actions', 'new_obs'],
        'latent_encoder': '../Model/checkpoints/latent_encoder',
        'num_context': [5, 10],
    }
    train_data_generator = CNPDataGenerator(generator_config)
    vali_data_generator = CNPDataGenerator(generator_config)

    # ---- build mlp model for latent states ---- #
    config = {
        'state_dims': 7,
        'action_dims': 3,
        'logits_dims': 8,
        'latent_encoder': {LayerConfig(fc=8, activation='relu')},
        'latent_decoder': {LayerConfig(fc=8, activation='relu')},
    }

    cnp_model = build_cnp_model(config)
    cnp_model.compile(
        # run_eagerly=True,
        optimizer=keras.optimizers.Adam(5e-4),
        loss={'dist_concat': dist_logp},
        metrics={'dist_concat': dist_mse, 'mu': stats},
    )

    print(cnp_model.summary())
    keras.utils.plot_model(cnp_model, show_shapes=True, to_file=f'../Model/{cnp_model.name}.png')

    # ----------------- train ----------------- #
    cnp_model.load_weights(f'../Model/checkpoints/{cnp_model.name}.h5')
    cnp_model.fit(
        train_data_generator, epochs=100, steps_per_epoch=100,
        validation_data=vali_data_generator, validation_steps=20,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                f'../Model/checkpoints/{cnp_model.name}.h5', save_best_only=True,
                monitor='val_dist_concat_loss'),
            keras.callbacks.EarlyStopping(
                monitor='val_dist_concat_loss',
                patience=5, mode='auto', restore_best_weights=True),
            # keras.callbacks.TensorBoard()
        ],
    )
    # cnp_model.save(f'../Model/checkpoints/{cnp_model.name}.h5')
