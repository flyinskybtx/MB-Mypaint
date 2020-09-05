import tensorflow as tf
from ray.rllib.offline import JsonReader
from tensorflow import keras

from Env.core_config import *
from Env.windowed_env import WindowedCnnEnv
from Model.cnn_model import LayerConfig
from Model.windowd_cnn_model import WindowedCnnModel


def logits_loss_fn(y_true, y_pred):
    splited_logits = [logits for logits in tf.split(y_pred, [5, 5, 5], axis=1)]
    labels = [y for y in tf.split(y_true, 3, axis=1)]
    losses = [tf.losses.sparse_categorical_crossentropy(label, logits, from_logits=True)
              for i, (label, logits) in enumerate(zip(labels, splited_logits)) if i % 3 != 2]
    loss = tf.reduce_mean(losses)
    return loss


class DataGenerator(keras.utils.Sequence):
    def __init__(self, reader):
        self.reader = reader

    def __getitem__(self, index):
        batch = self.reader.next()
        logits = batch['actions']
        values = np.zeros((logits.shape[0], 1))
        return batch['obs'], {'logits': logits, 'values': values}

    def __len__(self):
        return 100000


if __name__ == '__main__':
    model_saved_name = 'supervised_window_model'

    env_config = {
        'image_size': experimental_config.image_size,
        'obs_size':experimental_config.obs_size,
        'window_size': experimental_config.window_size,
        'xy_size': experimental_config.image_size,
        'z_size': experimental_config.z_size,
        'brush_name': experimental_config.brush_name,
        'image_nums': experimental_config.image_nums,
        'action_shape': experimental_config.action_shape,
    }

    env = WindowedCnnEnv(env_config)
    obs = env.reset()
    observation_space = env.observation_space
    action_space = env.action_space
    num_outputs = np.sum(action_space.nvec)
    name = 'supervised_window_model'

    model_config = {
        'custom_model': name,
        "custom_model_config": {
            'blocks': [
                LayerConfig(conv=[64, (2, 2), 1], padding='same', batch_norm=False, activation='relu',
                            pool=[2, 2, 'same'], dropout=0.5),
                LayerConfig(conv=[32, (2, 2), 1], padding='same', batch_norm=False, activation='relu',
                            pool=[2, 2, 'same'], dropout=0.5),
                LayerConfig(conv=[16, (2, 2), 1], padding='same', activation='relu', dropout=0.5),
                LayerConfig(flatten=True),
                LayerConfig(fc=4096, activation='relu', dropout=0.5),
                LayerConfig(fc=2048, activation='relu', dropout=0.5),
                LayerConfig(fc=1024, activation='relu', dropout=0.5),
            ],
            'offline_dataset': '../Data/offline/windowed',
        },
    }

    model = WindowedCnnModel(observation_space, action_space, num_outputs, model_config, name)
    model.base_model.compile(
        loss={'logits': logits_loss_fn},
        optimizer=keras.optimizers.SGD(lr=1e-3))

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
