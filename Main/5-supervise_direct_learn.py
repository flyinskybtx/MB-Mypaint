import os
from datetime import datetime

from ray.rllib.evaluation import SampleBatchBuilder
from ray.rllib.offline import JsonWriter
from tensorflow import keras
import tensorflow as tf

tf.compat.v1.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None
)

from tqdm import tqdm
import numpy as np

from Data import DATA_DIR
from Data.HWDB.hwdb_data import DirectData
from Data.HWDB.load_HWDB import get_waypoints_samples, HWDB_DIR
from Env.direct_env import RobotDirectEnv
from Main import load_config
from Model import MODEL_DIR
from Model.callbacks import RobotDirectEnvVisCallback
from Model.policy_model import CustomPolicyModel
from results import LOG_DIR
from Model.keras_losses import make_direct_logits_loss_fn

if __name__ == '__main__':
    cfg = load_config()

    config = cfg.ray_config
    config.env_config = cfg.direct_env_config
    config.env_config.brush_config = cfg.brush_config

    env = RobotDirectEnv(config.env_config)

    # --- collect data
    strides = config.env_config.num_xy
    margin = 0
    author = '1002-c.pot'
    save_dir = os.path.join(DATA_DIR, 'HWDB/direct', author)
    if len(os.listdir(save_dir)) == 0:
        writer = JsonWriter(save_dir)
        batch_builder = SampleBatchBuilder()

        strokes = get_waypoints_samples(os.path.join(HWDB_DIR, author),
                                        margin, strides, config.env_config.num_waypoints)

        env.reset()

        for stroke in tqdm(strokes):
            zs = np.random.randint(0, 10, size=(config.env_config.num_waypoints, 1))
            action = np.concatenate([stroke, zs], axis=-1).tolist()

            obs, reward, _, _ = env.step(action)

            # --- Save
            batch_builder.add_values(
                obs=obs,
                actions=action,
                reward=reward,
            )
            writer.write(batch_builder.build_and_reset())

    # --- supervised train
    # policy_model_name = "direct_robot_policy"
    policy_model_name = "direct_mlp_policy"
    config.model = {
        'custom_model': policy_model_name,
        "custom_model_config": {
            # 'layers': cfg.policy_model_config.direct_cnn_layers,
            "layers": cfg.policy_model_config.direct_mlp_layers,
        }
    }

    model = CustomPolicyModel(env.observation_space, env.action_space, np.sum(env.action_space.nvec),
                              config.model, policy_model_name).base_model
    data_dirs = [save_dir]
    batch_size = 32
    data_loader = DirectData(data_dirs, batch_size)
    Xs, Ys = data_loader.get_all()
    model.compile(optimizer=keras.optimizers.Adam(5e-4),
                  loss={'logits': make_direct_logits_loss_fn(env.action_space.nvec, use_z=True),
                        'values': keras.losses.mse},
                  run_eagerly=True,
                  )
    weights_file = os.path.join(MODEL_DIR, f'checkpoints/{policy_model_name}.h5')
    if os.path.exists(weights_file):
        model.load_weights(weights_file)

    tb_log_dir = os.path.join(LOG_DIR, f'logs/{policy_model_name}/')
    os.makedirs(tb_log_dir, exist_ok=True)
    model.fit(Xs, Ys, epochs=100, batch_size=64,
              validation_split=0.2, shuffle=True,
              callbacks=[
                  keras.callbacks.EarlyStopping(
                      monitor='val_loss', patience=5, mode='auto', restore_best_weights=True),
                  keras.callbacks.TensorBoard(
                      log_dir=os.path.join(tb_log_dir, datetime.now().strftime("%m%d-%H%M")),
                      histogram_freq=1,
                      update_freq=1),
                  RobotDirectEnvVisCallback(data_loader,
                                            env.action_space.nvec,
                                            config.env_config.num_xy,
                                            config.env_config.num_z,
                                            config.env_config.obs_size,
                                            frequency=5,
                                            total=3),
                  keras.callbacks.ModelCheckpoint(filepath=os.path.join(MODEL_DIR,
                                                                        f'checkpoints/{policy_model_name}/'),
                                                  save_best_only=True,
                                                  save_weights_only=True,
                                                  save_freq=5,
                                                  monitor='loss'),
              ]
              )
    model.save_weights(weights_file)
