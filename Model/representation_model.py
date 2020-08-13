# TODO
from tensorflow import keras


def build_encoder(config):
    encoder = keras.Sequential(name='encoder')
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
                                       units=params)
        else:
            raise ValueError(f"Layer type is {_type} that is not defined")
        encoder.add(layer)

    encoder.add(keras.layers.Dense(name='bottleneck_layer',
                                   units=config['bottleneck'],
                                   activity_regularizer=keras.regularizers.l1(10e-5)))

    return encoder


def build_decoder(config):
    decoder = keras.Sequential(name='decoder')
    decoder.add(keras.layers.Input(name='decoder_input', shape=(config['bottleneck'], )))

    for i, (_type, params) in enumerate(config['decoder_layers'], start=len(config['encoder_layers']) + 1):
        if _type == 'dense':  # W=(N−1)∗S−2P+F
            layer = keras.layers.Dense(name=f'dense{i}',
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
    return decoder


def build_AE(config):
    inputs = keras.layers.Input(name='image_input', shape=(config['image_shape']))

    encoder = build_encoder(config)
    encoder.summary()

    decoder = build_decoder(config)
    decoder.summary()

    outputs = decoder(encoder(inputs))
    model = keras.Model(inputs, outputs)
    model.summary()
    return model, encoder, decoder


if __name__ == '__main__':
    config = {
        'image_shape': (64, 64, 3),
        'encoder_layers': [
            ('conv', [32, (2, 2), 1]),
            ('pool', [(2, 2), 2]),
            ('conv', [16, (2, 2), 2]),
            ('pool', [(2, 2), 2]),
            ('flatten', None),
            ('dense', 1024),
        ],
        'decoder_layers': [
            ('dense', 1024),
            ('dense', 1024),
            ('reshape', (8, 8, 16)),
            ('upsampling', (2, 2)),
            ('deconv', [32, (2, 2), 2]),
            ('upsampling', (2, 2)),
            ('deconv', [3, (2, 2), 1]),
        ],
        'bottleneck': 128,
    }

    # encoder = build_encoder(config)
    # encoder.summary()
    # 
    # decoder = build_decoder(config)
    # decoder.summary()
    build_AE(config)
