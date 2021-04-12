#
from ray.rllib.utils import try_import_tf

from Data.Deprecated.repr_model import ReprModel

tf1, tf, _version = try_import_tf()
# keras = tf.keras
from tensorflow import keras

from Model import MODEL_DIR
from Model import LayerConfig
from script.main_procs.hparams import define_hparams

if __name__ == '__main__':
    # Build AE network
    cfg = define_hparams()
    cfg.train_latent_encoder = True
    cfg.train_decoder = False

    cfg.encoder_layers = [
        LayerConfig(type='conv', filters=32, kernel_size=(2, 2), strids=1, activation='relu'),
        LayerConfig(type='pool', pool_size=2, strides=2),
        LayerConfig(type='dropout', rate=0.2),
        LayerConfig(type='conv', filters=16, kernel_size=(2, 2), strids=2, activation='relu'),
        LayerConfig(type='pool', pool_size=2, strides=2),
        LayerConfig(type='dropout', rate=0.2),
        LayerConfig(type='conv', filters=16, kernel_size=(2, 2), strids=2, activation='relu'),
        LayerConfig(type='pool', pool_size=2, strides=2),
        LayerConfig(type='dropout', rate=0.2),
        LayerConfig(type='flatten'),
        LayerConfig(type='dense', units=256, activation='relu'),

    ]
    cfg.decoder_layers = [
        LayerConfig(type='dense', units=256, activation='relu'),
        LayerConfig(type='dropout', rate=0.2),
        LayerConfig(type='dense', units=1024, activation='relu'),
        LayerConfig(type='dropout', rate=0.2),
        LayerConfig(type='reshape', target_shape=(8, 8, 16)),
        LayerConfig(type='upsampling', size=(2, 2)),
        LayerConfig(type='deconv', filters=16, kernel_size=(2, 2), strides=1, activation='relu'),
        LayerConfig(type='dropout', rate=0.2),
        LayerConfig(type='upsampling', size=(2, 2)),
        LayerConfig(type='deconv', filters=32, kernel_size=(2, 2), strides=1, activation='relu'),
        LayerConfig(type='dropout', rate=0.2),
        LayerConfig(type='upsampling', size=(2, 2)),
        LayerConfig(type='deconv', filters=1, kernel_size=(2, 2), strides=1, activation='sigmoid'),
    ]

    repr_model = ReprModel(cfg)
    repr_model.latent_encoder.summary()
    # repr_model.decoder.summary()

    keras.utils.plot_model(repr_model.latent_encoder, show_shapes=True,
                           to_file=f'{MODEL_DIR}/png/{repr_model.latent_encoder.name}.png')
    keras.utils.plot_model(repr_model.latent_decoder, show_shapes=True,
                           to_file=f'{MODEL_DIR}/png/{repr_model.latent_decoder.name}.png')
    # keras.utils.plot_model(repr_model.decoder, show_shapes=True,
    #                        to_file=f'{MODEL_DIR}/png/{repr_model.decoder.name}.png')
