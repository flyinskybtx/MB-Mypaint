import copy
from datetime import datetime

import numpy as np
import tensorflow as tf
from ray.rllib.evaluation import SampleBatchBuilder
from ray.rllib.models.tf.tf_action_dist import SquashedGaussian
from ray.rllib.offline import JsonWriter
from tensorflow import keras
from tqdm import tqdm

tf.compat.v1.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None
)

import os

from Data import DATA_DIR
from Data.HWDB.hwdb_data import WindowedData
from Data.HWDB.load_HWDB import get_skeleton_paths_from_pot, HWDB_DIR
from Data.data_process_lib import skeleton_path_to_wps, sample_actions_z
from Env.continuous_env import SimulatorContinuousEnv
from Main import load_config
from Model import MODEL_DIR
from Model.callbacks import ContinuousEnvVisCallback
from Model.keras_losses import gaussian_loss
from Model.policy_model import CustomPolicyModel
from results import LOG_DIR

if __name__ == '__main__':
    cfg = load_config()

    policy_model_name = "WindowModel"

    config = cfg.ray_config
    config.env_config = cfg.continuous_env_config
    config.env_config.brush_config = cfg.brush_config
    env = SimulatorContinuousEnv(config.env_config)

    # --- collect data
    margin = 0
    author = '1003-c.pot'
    save_dir = os.path.join(DATA_DIR, 'HWDB/continuous', author)
    os.makedirs(save_dir, exist_ok=True)
    if len(os.listdir(save_dir)) == 0:
        writer = JsonWriter(save_dir)
        batch_builder = SampleBatchBuilder()

        skeleton_paths = get_skeleton_paths_from_pot(os.path.join(HWDB_DIR, author),
                                                     margin, config.env_config.image_size, total_strokes=1000)

        for i, sp in enumerate(tqdm(skeleton_paths, desc="Execute and Record data：")):
            wps, action_xy = skeleton_path_to_wps(sp, config.env_config.xy_grid, config.env_config.image_size,
                                                  discrete=False)
            action_z = sample_actions_z(action_xy.shape[0], config.env_config.z_grid)
            actions = np.concatenate([action_xy, action_z], axis=-1)
            env.reset()
            start_point = sp[0][0]
            env.position = [start_point[0], start_point[1], 0]  # 重置出发点
            env.mypaint_painter.paint(*env.painter_position)  # 移动笔刷到出发点

            done = False
            for action in actions:
                if done:
                    break
                _, _, done, _ = env.step(action)

            target = np.copy(env.cur_img)
            if np.max(target) == 0:
                continue

            env.reset()
            start_point = sp[0][0]
            env.position = [start_point[0], start_point[1], 0]  # 重置出发点
            env.mypaint_painter.paint(*env.painter_position)  # 移动笔刷到出发点
            env.target_image = target

            for action in actions:
                env.step(action)

            new_obs = env.obs
            count = 0
            done = False
            for action in actions:
                if done:
                    break
                obs = copy.deepcopy(new_obs)
                new_obs, reward, done, _ = env.step(action)
                if done:
                    break

                if np.max(new_obs[:, :, 2]) > 0:
                    batch_builder.add_values(

                        obs=obs,
                        actions=action.tolist(),
                        new_obs=new_obs,
                        reward=reward,
                    )
                    count += 1
            if count > 0:
                batch = batch_builder.build_and_reset()
                writer.write(batch)

    # --- supervised train

    config.model = {
        'custom_model': policy_model_name,
        'layers': cfg.policy_model_config.direct_cnn_layers,
    }
    num_outputs = SquashedGaussian.required_model_output_shape(env.action_space, config.model)
    model = CustomPolicyModel(env.observation_space, env.action_space, num_outputs, config.model,
                              policy_model_name).base_model
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss={'logits': gaussian_loss,
                        'values': keras.losses.mse},
                  run_eagerly=True, )

    # --- Train ---
    # weights_checkpoint = os.path.join(MODEL_DIR, 'checkpoints/continuous/')
    # if os.path.exists(weights_checkpoint):
    #     model.load_weights(weights_checkpoint)

    weights_file = os.path.join(MODEL_DIR, f'checkpoints/{policy_model_name}.h5')
    if os.path.exists(weights_file):
        model.load_weights(weights_file)

    data_dirs = [save_dir]
    count = 32
    data_loader = WindowedData(data_dirs, count)
    Xs, Ys = data_loader.get_all(10000)

    tb_log_dir = os.path.join(LOG_DIR, f'logs/{policy_model_name}/')
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
                  keras.callbacks.ModelCheckpoint(
                      filepath=os.path.join(MODEL_DIR, f'checkpoints/{policy_model_name}/'),
                      save_best_only=True,
                      save_weights_only=True,
                      save_freq=5,
                      monitor='loss'),
              ]
              )
    model.save_weights(weights_file)
