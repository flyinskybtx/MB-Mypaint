import copy
import os.path as osp

import numpy as np
#
from ray.rllib.utils import try_import_tf

from Data.Deprecated.auto_encoder_data import AEGenerator

tf1, tf, _version = try_import_tf()
from tensorflow import keras
from Model import MODEL_DIR, make_layer


def build_obs_encoder(config):
    inputs = keras.layers.Input(name='Encoder_input', shape=(config.obs_size, config.obs_size, 1))
    x = inputs
    for i, layer_config in enumerate(config.encoder_layers, start=1):
        layer_config.number = i
        layer = make_layer(layer_config)
        x = layer(x)

    if config.is_vae:
        mu = keras.layers.Dense(config.latent_size, name='Vae_mu')(x)
        sigma = keras.layers.Dense(config.latent_size, name='Vae_sigma')(x)
        outputs = [mu, sigma]
        name = 'Vae_encoder'
    else:
        outputs = keras.layers.Dense(config.latent_size, name='Latent_layer')(x)
        name = 'MLP_encoder'

    encoder = keras.models.Model(inputs, outputs, name=name)
    return encoder


def build_obs_decoder(config):
    inputs = keras.layers.Input(name='Latent_input', shape=(config.latent_size,))
    x = inputs
    if config.is_vae:
        name = 'Vae_decoder'
    else:
        name = 'MLP_decoder'

    for i, layer_config in enumerate(config.decoder_layers, start=1):
        layer_config.number = i
        layer = make_layer(layer_config)
        x = layer(x)

    assert config.decoder_layers[-1].activation == 'sigmoid'
    if config.is_bw_output:
        outputs = keras.layers.Lambda(lambda x: round_through(x), name='Bw_output')(x)
    else:
        outputs = x
    decoder = keras.models.Model(inputs, outputs, name=name)
    return decoder


def build_decoder(latent_encoder, latent_decoder, config):
    cur_encoder = keras.models.clone_model(latent_encoder)
    cur_encoder._name = 'Cur_encoder'
    cur_encoder.layers[0]._name = 'Cur_inputs'
    cur_inputs = cur_encoder.inputs
    cur_encoder.trainable = False

    latent_inputs = keras.layers.Input(name='Latent_input', shape=(config.latent_size,))
    x = latent_inputs

    for layer in cur_encoder.layers:  # Freeze obs_encoder layers
        layer.trainable = False
    encoder_layers = {layer.name: layer.output for layer in cur_encoder.layers}
    latent_decoder_weights = {layer.name: layer.get_weights() for layer in latent_decoder.layers}

    for i, layer_config in enumerate(config.decoder_layers, start=1):
        layer_config.number = i
        layer = make_layer(layer_config)
        x = layer(x)
        if layer.name in latent_decoder_weights.keys():  # Set weights to trained latent decoder if exists
            layer.set_weights(latent_decoder_weights[layer.name])
        if layer_config.type == 'upsampling':
            skip_connection = encoder_layers[f'Conv{len(config.decoder_layers) - i}']
            x = keras.layers.Add(name=f'SkipConn{i}')([x, skip_connection])
    outputs = x
    # outputs = keras.layers.Lambda(lambda xs: round_through(xs), name='Bw_output')(xs)
    decoder = keras.models.Model(inputs=[cur_inputs, latent_inputs], outputs=outputs, name='Decoder')
    return decoder


class ReprModel:
    def __init__(self, config):
        self.config = config

        self.latent_encoder = build_obs_encoder(config)
        self.latent_decoder = build_obs_decoder(config)
        if config.use_skip_conn:
            self.decoder = build_decoder(self.latent_encoder, self.latent_decoder, config)
            if self.config.train_decoder:
                self.train_decoder()
            else:
                self.load_decoder()

        if self.config.train_latent_encoder:
            self.train_latent_encoder()
        else:
            self.load_latent_encoder()

    def train_mlp(self, data_dir='offline/random', batch_size=16):
        inputs = self.latent_encoder.inputs
        outputs = self.latent_decoder(self.latent_encoder(inputs))
        latent_auto_encoder = keras.models.Model(inputs, outputs, name='LatentAutoEncoder')
        latent_auto_encoder.summary()
        keras.utils.plot_model(latent_auto_encoder, show_shapes=True,
                               to_file=f'{MODEL_DIR}/png/{latent_auto_encoder.name}.png')

        # Implement data_loader loader
        train_data = AEGenerator(savedir=data_dir, batch_size=batch_size, is_encoder=True, max_samples=1e5)
        vali_data = copy.deepcopy(train_data)
        vis_data = copy.deepcopy(train_data)

        # Train AE network
        latent_auto_encoder.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(lr=5e-3),
        )
        records = latent_auto_encoder.fit(
            train_data, epochs=100, steps_per_epoch=min(1000, train_data.__len__()),
            validation_data=vali_data, validation_steps=200,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_mse', patience=5, mode='auto', restore_best_weights=True),
                VisGenCallback(vis_data, frequency=5)
            ]
        )

    def train_latent_encoder(self, data_dir='offline/random', batch_size=16):
        if self.config.is_vae:
            self.train_vae(data_dir, batch_size)
        else:
            self.train_mlp(data_dir, batch_size)

        self.latent_encoder.save_weights(osp.join(MODEL_DIR, f'checkpoints/{self.latent_encoder.name}.h5'))
        self.latent_decoder.save_weights(osp.join(MODEL_DIR, f'checkpoints/{self.latent_decoder.name}.h5'))

    def train_decoder(self, data_dir='offline/random', batch_size=16):
        next_encoder = keras.models.clone_model(self.latent_encoder)
        next_encoder._name = 'Next_encoder'
        next_encoder.layers[0]._name = 'Next_inputs'
        next_inputs = next_encoder.inputs
        next_encoder.trainable = False

        cur_inputs = self.decoder.inputs[0]
        latent = next_encoder(next_inputs)
        outputs = self.decoder([cur_inputs, latent])

        auto_encoder = keras.models.Model(inputs=[cur_inputs, next_inputs], outputs=outputs, name='AutoEncoder')
        auto_encoder.summary()
        keras.utils.plot_model(auto_encoder, show_shapes=True, to_file=f'{MODEL_DIR}/png/{auto_encoder.name}.png')

        train_data = AEGenerator(savedir=data_dir, batch_size=batch_size, is_encoder=False)
        vali_data = AEGenerator(savedir=data_dir, batch_size=batch_size, is_encoder=False)
        vis_data = AEGenerator(savedir=data_dir, batch_size=batch_size, is_encoder=False)

        auto_encoder.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(lr=5e-3),
        )
        records = auto_encoder.fit(
            train_data, epochs=100, steps_per_epoch=min(1000, train_data.__len__()),
            validation_data=vali_data, validation_steps=200,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=5, mode='auto', restore_best_weights=True),
                VisGenCallback(vis_data, frequency=5)
            ]
        )
        self.decoder.save_weights(osp.join(MODEL_DIR, f'checkpoints/{self.decoder.name}.h5'))

    def load_latent_encoder(self):
        self.latent_encoder.load_weights(osp.join(MODEL_DIR, f'checkpoints/{self.latent_encoder.name}.h5'))
        self.latent_decoder.load_weights(osp.join(MODEL_DIR, f'checkpoints/{self.latent_decoder.name}.h5'))
        self.latent_encoder.make_predict_function()
        self.latent_decoder.make_predict_function()
        print(f"Load latent encoder from {self.latent_encoder.name}.h5")
        print(f"Load latent obs_decoder from {self.latent_encoder.name}.h5")

    def load_decoder(self):
        self.decoder.load_weights(osp.join(MODEL_DIR, f'checkpoints/{self.decoder.name}.h5'))
        self.decoder.make_predict_function()
        print(f"Load obs_decoder from {self.decoder.name}.h5")

    def latent_encode(self, obs):
        if len(obs.shape) == 3:
            obs = np.expand_dims(obs, axis=0)
        delta = np.expand_dims(obs[:, :, :, 0] - obs[:, :, :, 1], axis=-1)
        return self.latent_encoder.predict(delta)

    def latent_decode(self, latent):
        if isinstance(latent, list):
            delta = self.latent_decoder.predict(latent)
            return delta

        if len(latent.shape) == 1:
            latent = np.expand_dims(latent, axis=0)
        delta = self.latent_decoder.predict(latent)
        return delta
