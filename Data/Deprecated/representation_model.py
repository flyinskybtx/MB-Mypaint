import tensorflow as tf
from tensorflow import keras

from Data.Deprecated.core_config import experimental_config


def round_through(x):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    '''
    rounded = tf.round(x)
    return x + tf.stop_gradient(rounded - x)


def build_encoder(config):
    encoder = keras.Sequential(name='latent_encoder')
    encoder.add(keras.layers.Input(name='encoder_input', shape=(config['image_shape'])))
    for i, (_type, params) in enumerate(config['encoder_layers'], start=1):
        if _type == 'conv':  # N=(W−F+2P)/S+1
            layer = keras.layers.Conv2D(name=f'conv{i}',
                                        filters=params[0],
                                        kernel_size=params[1],
                                        strides=params[2],
                                        padding='same',
                                        activation='relu')
        elif _type == 'pool':
            layer = keras.layers.MaxPooling2D(name=f'pool{i}',
                                              pool_size=params[0],
                                              strides=params[1],
                                              padding='same')
        elif _type == 'flatten':
            layer = keras.layers.Flatten(name=f'flatten{i}')
        elif _type == 'dense':
            layer = keras.layers.Dense(name=f'dense{i}',
                                       activation='relu',
                                       units=params)
        else:
            raise ValueError(f"Layer type is {_type} that is not defined")
        encoder.add(layer)

    encoder.add(keras.layers.Dense(name='bottleneck_layer',
                                   units=config['bottleneck'],
                                   activation='linear',
                                   activity_regularizer=keras.regularizers.l1(10e-5)))

    return encoder


def build_decoder(config):
    decoder = keras.Sequential(name='latent_decoder')
    decoder.add(keras.layers.Input(name='decoder_input', shape=(config['bottleneck'],)))

    for i, (_type, params) in enumerate(config['decoder_layers'], start=len(config['encoder_layers']) + 1):
        if _type == 'dense':  # W=(N−1)∗S−2P+F
            layer = keras.layers.Dense(name=f'dense{i}',
                                       activation='relu',
                                       units=params)
        elif _type == 'reshape':
            layer = keras.layers.Reshape(name=f'reshape{i}', target_shape=params)
        elif _type == 'upsampling':
            layer = keras.layers.UpSampling2D(name=f'upsample{i}', size=params)
        elif _type == 'deconv':
            layer = keras.layers.Conv2DTranspose(name=f'deconv{i}',
                                                 filters=params[0],
                                                 kernel_size=params[1],
                                                 strides=params[2],
                                                 padding='same',
                                                 activation='relu')
        else:
            raise ValueError(f"Layer type is {_type} that is not defined")
        decoder.add(layer)
    decoder.add(keras.layers.Lambda(lambda x: round_through(x), name='bw_output'))

    return decoder


def build_AE(config):
    inputs = keras.layers.Input(name='image_input', shape=(config['image_shape']))

    encoder = build_encoder(config)
    encoder.summary()
    keras.utils.plot_model(encoder, show_shapes=True, to_file=f'../Model/{encoder.name}.png')

    decoder = build_decoder(config)

    decoder.summary()
    keras.utils.plot_model(decoder, show_shapes=True, to_file=f'../Model/{decoder.name}.png')
    outputs = decoder(encoder(inputs))
    model = keras.Model(inputs, outputs)
    model.summary()
    return model, encoder, decoder


def build_state_AE(config):
    # build latent_encoder
    cur = keras.layers.Input(name='encoder_cur', shape=(config['bottleneck'],))
    prev = keras.layers.Input(name='encoder_prev', shape=(config['bottleneck'],))
    x = keras.layers.Concatenate(name='latent_concat', axis=-1)([cur, prev])
    for i, (_type, params) in enumerate(config['encoder_layers'], start=1):
        if _type == 'dense':
            x = keras.layers.Dense(name=f'dense{i}', units=params, activation='relu')(x)
    x = keras.layers.Dense(name='state', units=config['state_dim'],
                           activity_regularizer=keras.regularizers.l1(10e-5))(x)
    encoder = keras.models.Model(inputs=[cur, prev], outputs=x, name='state_encoder')
    encoder.summary()

    # build latent_decoder
    decoder = keras.models.Sequential(name='state_decoder')
    decoder.add(keras.layers.Input(name='state', shape=(config['state_dim'],)))
    for i, (_type, params) in enumerate(config['encoder_layers'], start=1):
        if _type == 'dense':
            decoder.add(keras.layers.Dense(name=f'dense{i}', units=params, activation='relu'))
    decoder.add(keras.layers.Dense(name='delta_latent', units=config['bottleneck']))
    decoder.summary()

    # as a whole
    input_cur = keras.layers.Input(name='cur_latent', shape=(config['bottleneck'],))
    input_prev = keras.layers.Input(name='prev_latent', shape=(config['bottleneck'],))
    inputs = [input_cur, input_prev]
    outputs = decoder(encoder(inputs))
    model = keras.Model(inputs, outputs)
    model.summary()
    return model, encoder, decoder


if __name__ == '__main__':
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
    state_AE_config = {
        'encoder_layers': [
            ('dense', 32),
            ('dense', 16),
        ],
        'decoder_layers': [
            ('dense', 16),
            ('dense', 32),
        ],
        'bottleneck': 32,
        'state_dim': 7,
    }

    # latent_encoder = build_encoder(config)
    # latent_encoder.summary()
    # 
    # latent_decoder = build_decoder(config)
    # latent_decoder.summary()
    build_AE(config)
    build_state_AE(state_AE_config)
