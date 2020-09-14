import numpy as np
import tensorflow as tf
from ray.rllib.offline import JsonReader
from tensorflow import keras

from Data.Deprecated.core_config import experimental_config
from Env.direct_env import DirectCnnEnv
from Model.cnn_model import LayerConfig
from Model.supervised_cnn_model import SupervisedCnnModel
from script.tests.train_supervised_windowed import DataGenerator


def make_logits_loss(input_lens):
    def logits_loss_fn(y_true, y_pred):
        splited_logits = [logits for logits in tf.split(y_pred, input_lens, axis=1)]
        labels = [y for y in tf.split(y_true, len(input_lens), axis=1)]
        losses = [tf.losses.sparse_categorical_crossentropy(label, logits, from_logits=True)
                  for i, (label, logits) in enumerate(zip(labels, splited_logits)) if i % 3 != 2]

        loss = tf.reduce_mean(losses)
        return loss

    return logits_loss_fn


def logits_metric(y_true, y_pred):
    pass


if __name__ == '__main__':
    model_saved_name = 'supervised_direct_model'

    env_config = {
        'image_size': experimental_config.image_size,
        'stride_size': experimental_config.stride_size,
        'stride_amplify': experimental_config.stride_amplify,
        'z_size': experimental_config.z_size,
        'brush_name': experimental_config.brush_name,
        'num_keypoints': experimental_config.num_keypoints,
        'image_nums': experimental_config.image_nums,
    }

    env = DirectCnnEnv(env_config)
    obs = env.reset()
    observation_space = env.observation_space
    action_space = env.action_space
    num_outputs = np.sum(action_space.nvec)
    name = 'test_custom_cnn'

    model_config = {
        'custom_model': name,
        "custom_model_config": {
            'blocks': [
                LayerConfig(conv=[128, (2, 2), 1], padding='same', batch_norm=False, activation='relu',
                            pool=[2, 2, 'same'], dropout=0.5),
                LayerConfig(conv=[64, (2, 2), 1], padding='same', batch_norm=False, activation='relu',
                            pool=[2, 2, 'same'], dropout=0.5),
                LayerConfig(conv=[32, (2, 2), 1], padding='same', activation='relu', dropout=0.5),
                LayerConfig(flatten=True),
                LayerConfig(fc=4096, activation='relu', dropout=0.5),
                LayerConfig(fc=2048, activation='relu', dropout=0.5),
                LayerConfig(fc=1024, activation='relu', dropout=0.5),
            ],
            'offline_dataset': '../Data/offline/direct'
        },
    }

    model = SupervisedCnnModel(observation_space, action_space, num_outputs, model_config, name)
    logits_loss_fn = make_logits_loss(input_lens=model.action_space.nvec)

    model.base_model.compile(
        loss={'logits': logits_loss_fn},
        optimizer=keras.optimizers.Adam(lr=5e-4, epsilon=1e-5))

    reader = JsonReader(model.model_config['custom_model_config']['offline_dataset'])
    data_generator = DataGenerator(reader)

    model.base_model.fit_generator(
        data_generator, epochs=100, steps_per_epoch=100,
        validation_data=data_generator, validation_steps=100,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                f'../Model/checkpoints/{model_saved_name}.h5', save_best_only=True),
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, mode='auto', restore_best_weights=True),
        ]
    )
    # Examine

    model.base_model.load_weights(f'../Model/checkpoints/{model_saved_name}.h5')
