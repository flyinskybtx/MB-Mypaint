import os
from datetime import datetime
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy
from skimage import transform
from tensorflow import keras

from Data import DATA_DIR
from Data.HWDB.hwdb_data import DirectData
from Env.direct_env import DirectEnv
from Model import LayerConfig, MODEL_DIR
from Model.keras_losses import make_direct_logits_loss_fn
from Model.policy_model import CustomPolicyModel
from results import LOG_DIR
from script.main_procs.hparams import define_hparams


class DirectEnvVisCallback(keras.callbacks.Callback):
    def __init__(self, data_loader, nvec, stride_size, z_grid, frequency=5, total=10, img_dir=None):
        super().__init__()
        self.data_loader = data_loader
        self.shuffle = data_loader.shuffle
        self.frequency = frequency
        self.total = total
        self.img_dir = img_dir
        self.stride_size = stride_size
        self.z_grid = z_grid
        self.nvec = nvec
        self.image_size = nvec[0] * self.stride_size

    def action_to_frame(self, action):
        action = action.reshape(-1, 3)
        wps = action * np.array([self.stride_size, self.stride_size, self.z_grid]) + np.array(
            [self.stride_size / 2, self.stride_size / 2, 0])

        wps_frame = np.zeros((self.image_size, self.image_size))

        for wp in wps:
            wps_frame[int(wp[0]), int(wp[1])] = wp[2]
        kernel = np.ones((self.stride_size, self.stride_size))
        wps_frame = scipy.ndimage.convolve(wps_frame, kernel, mode='constant', cval=0.0)
        return wps_frame

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.frequency != 0:
            return

        obs, (y_actions, _) = self.data_loader.__getitem__(0)
        logits, _ = self.model(obs)
        logits = logits.numpy()

        if self.img_dir is not None:
            os.makedirs(self.img_dir, exist_ok=True)

        for img, logit, y_action in zip(obs[:self.total], logits[:self.total], y_actions[:self.total]):
            splited = np.split(logit, np.cumsum(self.nvec))[:-1]  # split return N+1 values
            action = np.array([np.argmax(l) for l in splited])

            wps_pred = self.action_to_frame(action)
            wps_true = self.action_to_frame(y_action)

            wps_pred = np.clip(transform.resize(wps_pred, img.shape), 0, 1)
            wps_true = np.clip(transform.resize(wps_true, img.shape), 0, 1)

            plt.imshow(np.concatenate([wps_pred, wps_true, img], axis=-1))
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

    env = DirectEnv(env_config)
    model = CustomPolicyModel(env.observation_space, env.action_space, np.sum(env.action_space.nvec), model_config,
                           'dummy_env').base_model
    model.summary()

    data_dirs = glob(os.path.join(DATA_DIR, 'HWDB/json/*'))
    batch_size = 32
    data_loader = DirectData(data_dirs, batch_size)
    Xs, Ys = data_loader.get_all()

    model.compile(optimizer=keras.optimizers.Adam(5e-4),
                  loss={'logits': make_direct_logits_loss_fn(env.action_space.nvec, use_z=True),
                        'values': keras.losses.mse},
                  run_eagerly=True,
                  )

    # --- Train ---
    weights_file = os.path.join(MODEL_DIR, 'checkpoints/direct_policy.h5')
    if os.path.exists(weights_file):
        model.load_weights(weights_file)

    tb_log_dir = os.path.join(LOG_DIR, 'logs/direct_/')
    os.makedirs(tb_log_dir, exist_ok=True)
    model.fit(Xs, Ys, epochs=100, batch_size=64,
              validation_split=0.2, shuffle=True,
              callbacks=[
                  keras.callbacks.EarlyStopping(
                      monitor='loss', patience=5, mode='auto', restore_best_weights=True),
                  keras.callbacks.TensorBoard(
                      log_dir=os.path.join(tb_log_dir, datetime.now().strftime("%m%d-%H%M")),
                      histogram_freq=1,
                      update_freq=1),
                  DirectEnvVisCallback(data_loader, nvec=env.action_space.nvec,
                                       stride_size=env_config.stride_size,
                                       z_grid=env_config.z_grid, frequency=5, total=3),
                  keras.callbacks.ModelCheckpoint(filepath=os.path.join(MODEL_DIR,
                                                                        'checkpoints/direct_policy/'),
                                                  save_best_only=True,
                                                  save_weights_only=True,
                                                  save_freq=5,
                                                  monitor='loss'),
              ]
              )
    model.save_weights(weights_file)
