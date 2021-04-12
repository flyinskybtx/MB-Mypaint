import os
from datetime import datetime
from glob import glob

from tensorflow import keras

from Data import DATA_DIR
from ray.rllib.models.tf.tf_action_dist import SquashedGaussian

from Data.HWDB.hwdb_data import WindowedData
from Env.continuous_env import ContinuousEnv
from Model import LayerConfig, MODEL_DIR
from Model.callbacks import ContinuousEnvVisCallback
from Model.keras_losses import gaussian_loss
from Model.policy_model import CustomPolicyModel
from results import LOG_DIR
from script.main_procs.hparams import define_hparams

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

    env = ContinuousEnv(env_config)
    num_outputs = SquashedGaussian.required_model_output_shape(env.action_space, model_config)
    model = CustomPolicyModel(env.observation_space, env.action_space, num_outputs, model_config, 'dummy_env').base_model
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss={'logits': gaussian_loss,
                        'values': keras.losses.mse},
                  run_eagerly=True,
                  )

    # --- Train ---
    weights_checkpoint = os.path.join(MODEL_DIR, 'checkpoints/continuous/')
    if os.path.exists(weights_checkpoint):
        model.load_weights(weights_checkpoint)

    weights_file = os.path.join(MODEL_DIR, 'checkpoints/continuous_policy.h5')
    if os.path.exists(weights_file):
        model.load_weights(weights_file)

    data_dirs = glob(os.path.join(DATA_DIR, 'HWDB/continuous/*'))
    batch_size = 32
    data_loader = WindowedData(data_dirs, batch_size)
    Xs, Ys = data_loader.get_all(10000)

    tb_log_dir = os.path.join(LOG_DIR, 'logs/continuous_/')
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
                  ContinuousEnvVisCallback(data_loader, frequency=5, total=3),
                  keras.callbacks.ModelCheckpoint(filepath=os.path.join(MODEL_DIR,
                                                                        'checkpoints/continuous_policy/'),
                                                  save_best_only=True,
                                                  save_weights_only=True,
                                                  save_freq=5,
                                                  monitor='loss'),
              ]
              )
    model.save_weights(weights_file)
