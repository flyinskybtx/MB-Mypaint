import os
from datetime import datetime
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from Data import DATA_DIR
from Data.HWDB.hwdb_data import WindowedData
from Env.discrete_env import DiscreteEnv
from Model import LayerConfig, MODEL_DIR
from Model.policy_model import CustomPolicyModel
from results import LOG_DIR
from script.main_procs.hparams import define_hparams


def make_logits_loss_fn(nvec, use_z=False):
    nvec = nvec.tolist()
    if use_z:
        mask = np.array([True, True, False] * (len(nvec) // 3))

    def logits_loss(y_true, y_pred):
        splited_logits = [logits for logits in tf.split(y_pred, nvec, axis=1)]
        labels = [y for y in tf.split(y_true, len(nvec), axis=1)]
        losses = [tf.losses.sparse_categorical_crossentropy(label, logits, from_logits=True) for
                  label, logits in zip(labels, splited_logits)]

        if not use_z:
            losses = tf.boolean_mask(losses, mask)

        loss = tf.reduce_mean(losses)
        return loss

    return logits_loss


class DiscreteEnvVisCallback(keras.callbacks.Callback):
    def __init__(self, data_loader, nvec, frequency=5, total=10):
        super().__init__()
        self.data_loader = data_loader
        self.frequency = frequency
        self.total = total
        self.nvec = nvec

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.frequency != 0:
            return

        i = np.random.randint(0, self.data_loader.__len__())
        obs, (action_true, _) = self.data_loader.__getitem__(i)
        logits, _ = self.model(obs)
        logits = logits.numpy()

        for img, logit, y_action in zip(obs[:self.total], logits[:self.total], action_true[:self.total]):
            splited = np.split(logit, np.cumsum(self.nvec))[:-1]  # split return N+1 values
            action_pred = np.array([np.argmax(l) for l in splited])

            plt.imshow(img[:, :, :3])
            plt.suptitle(f'Action: {action_pred}')
            plt.show()


if __name__ == '__main__':
    # Settings
    env_config = define_hparams()
    physics = glob(os.path.join(DATA_DIR, f'offline/slow_ink/Physics'))[0]
    env_config.brush_info_file = os.path.join(physics, 'BrushInfo.myb')

    model_config = {
        'custom_model': 'supervised',
        "custom_model_config": {
            'layers': [
                LayerConfig(type='batch_norm'),
                LayerConfig(type='conv', filters=32, kernel_size=(2, 2), strids=1, activation='relu'),
                LayerConfig(type='pool', pool_size=2, strides=2),
                LayerConfig(type='batch_norm'),
                LayerConfig(type='conv', filters=16, kernel_size=(2, 2), strids=2, activation='relu'),
                LayerConfig(type='pool', pool_size=2, strides=2),
                LayerConfig(type='batch_norm'),
                LayerConfig(type='conv', filters=16, kernel_size=(2, 2), strids=2, activation='relu'),
                LayerConfig(type='pool', pool_size=2, strides=2),
                LayerConfig(type='flatten'),
                LayerConfig(type='dropout', rate=0.2),
                LayerConfig(type='dense', units=1024, activation='relu'),
            ]
        },
    }

    env = DiscreteEnv(env_config)
    model = CustomPolicyModel(env.observation_space, env.action_space, np.sum(env.action_space.nvec), model_config,
                           'dummy_env').base_model
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(5e-4),
                  loss={'logits': make_logits_loss_fn(env.action_space.nvec, use_z=True),
                        'values': keras.losses.mse},
                  run_eagerly=True,
                  )

    # --- Train ---
    model.load_weights(os.path.join(MODEL_DIR, 'checkpoints/discrete_policy/'))

    weights_file = os.path.join(MODEL_DIR, 'checkpoints/discrete_policy.h5')
    if os.path.exists(weights_file):
        model.load_weights(weights_file)

    data_dirs = glob(os.path.join(DATA_DIR, 'HWDB/discrete/*'))
    batch_size = 32
    data_loader = WindowedData(data_dirs, batch_size)
    Xs, Ys = data_loader.get_all(10000)

    tb_log_dir = os.path.join(LOG_DIR, 'logs/discrete_/')
    os.makedirs(tb_log_dir, exist_ok=True)
    model.fit(Xs, Ys, epochs=100, batch_size=32,
              validation_split=0.2, shuffle=True,
              callbacks=[
                  keras.callbacks.EarlyStopping(
                      monitor='val_loss', patience=5, mode='auto', restore_best_weights=True),
                  keras.callbacks.TensorBoard(
                      log_dir=os.path.join(tb_log_dir, datetime.now().strftime("%m%d-%H%M")),
                      histogram_freq=1,
                      update_freq=1),
                  DiscreteEnvVisCallback(data_loader, nvec=env.action_space.nvec, frequency=5, total=3),
                  keras.callbacks.ModelCheckpoint(filepath=os.path.join(MODEL_DIR,
                                                                        'checkpoints/discrete_policy/'),
                                                  save_best_only=True,
                                                  save_weights_only=True,
                                                  save_freq=5,
                                                  monitor='loss'),
              ]
              )
    model.save_weights(weights_file)
