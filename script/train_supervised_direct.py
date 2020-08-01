import numpy as np
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os.path as osp

from Data.data_process import load_stroke_png, preprocess_stroke_png, extract_skeleton_trace, \
    get_supervised_wps_from_track
from Env.core_config import *
from Env.direct_env import DirectCnnEnv
from Model.cnn_model import LayerConfig
from Model.supervised_cnn_model import SupervisedCnnModel


def make_logits_loss(model_config, input_lens):
    mask = np.array([True, True, False] * (len(input_lens) // 3))

    def logits_loss_fn(y_true, y_pred):
        splited_logits = [logits for logits in tf.split(y_pred, input_lens, axis=1)]
        labels = [y for y in tf.split(y_true, len(input_lens), axis=1)]
        losses = [tf.losses.sparse_categorical_crossentropy(label, logits, from_logits=True) for i, (label, logits) in
                  enumerate(zip(labels,
                                splited_logits))
                  if i % 3 != 2]

        loss = tf.reduce_mean(losses)
        # action_dist = MultiCategorical(y_pred, model_config, input_lens=input_lens)
        # logp = -action_dist.logp(y_true)
        # masked_logp = tf.boolean_mask(logp, mask)
        # loss = tf.reduce_mean(masked_logp)
        return loss

    return logits_loss_fn


def logits_metric(y_true, y_pred):
    pass


# Settings

image_size = IMAGE_SIZE
roi_grid_size = ROI_GRID_SIZE
pixels_per_grid = PIXELS_PER_GRID
z_grid_size = Z_GRID_SIZE
num_keypoints = NUM_KEYPOINTS
image_name = IMAGE_NAME

ori_img = load_stroke_png(image_name)
print(f'Shape of origin image is {ori_img.shape}')

preprocessed_img = preprocess_stroke_png(ori_img, image_size=image_size)
print(f'Shape of preprocessed image is {preprocessed_img.shape}')

reference_path = extract_skeleton_trace(preprocessed_img, roi_grid_size, discrete=True)
supervised_path = get_supervised_wps_from_track(reference_path, num_keypoints)

model_config = {
    'custom_model': 'model_name',
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
        'supervised_action': supervised_path,
    },
}

env_config = {
    'image_size': image_size,
    'roi_grid_size': roi_grid_size,
    'pixels_per_grid': pixels_per_grid,
    'z_grid_size': z_grid_size,
    'brush_name': 'custom/slow_ink',
    'num_keypoints': num_keypoints,
    'target_image': preprocessed_img,
}

if __name__ == '__main__':
    env = DirectCnnEnv(env_config)
    obs = env.reset()
    observation_space = env.observation_space
    action_space = env.action_space
    num_outputs = np.sum(action_space.nvec)
    name = 'test_custom_cnn'

    model = SupervisedCnnModel(observation_space, action_space, num_outputs, model_config, name)
    logits_loss_fn = make_logits_loss(model.model_config, model.action_space.nvec)

    model.base_model.compile(
        loss={'logits': logits_loss_fn},
        optimizer=keras.optimizers.SGD(lr=1e-3))

    img = np.zeros(shape=(int(image_size / roi_grid_size * pixels_per_grid),
                          int(image_size / roi_grid_size * pixels_per_grid),
                          1))

    imgs = np.array([obs])
    logits = np.array([supervised_path.reshape(-1, )])
    values = np.array([np.zeros(1)])
    model.base_model.fit(x=imgs, y={'logits': logits, 'values': values}, epochs=100)
    model.base_model.save('../Model/checkpoints/supervised_model.h5')
    # Examine

    model.base_model.load_weights('../Model/checkpoints/supervised_model.h5')
