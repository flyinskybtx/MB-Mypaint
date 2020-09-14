# Rewards:
import numpy as np
from fastdtw import fastdtw
from skimage.metrics import mean_squared_error
from sklearn import metrics


def img_cosine_reward(tar, obs):
    value = metrics.pairwise.cosine_similarity(tar.reshape(1, -1), obs.reshape(1, -1))[0, 0]
    if value == np.nan:
        raise ValueError
        # return -1
    return value


def img_mse_loss(tar, obs):
    return - mean_squared_error(tar, obs)


def scale_loss(tar, obs):
    if np.sum(obs) == 0:  # If observation is empty img, accu_reward 0
        return -1
    else:
        tar_h, tar_w = map(lambda x: np.max(x) - np.min(x), np.where(tar > 0))
        obs_h, obs_w = map(lambda x: np.max(x) - np.min(x), np.where(obs > 0))
        img_shape = tar.shape
        return - 0.5 * (np.abs(tar_h - obs_h) / img_shape[0] + np.abs(tar_w - obs_w) / img_shape[1])


def iou_reward(tar, obs):
    """ iou between image patterns, all images must be BW-image """
    value = np.count_nonzero(tar + obs == 2) / np.count_nonzero(tar + obs > 0)
    return value


def curvature_loss(ref_path, cur_path):
    def _calc_path_curvature(waypoints: np.ndarray):
        """ calculate the curvature of 2-D waypoints """
        waypoints = waypoints.copy()
        dx_dt = np.gradient(waypoints[:, 0])
        dy_dt = np.gradient(waypoints[:, 1])

        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)

        curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2 + 1e-6) / (dx_dt * dx_dt + dy_dt * dy_dt + 1e-6) ** 1.5
        curvature = np.clip(curvature, 0, 1)
        return curvature

    if len(cur_path) < 2:  # if path too short, return a big loss
        return 1

    ref_curve = _calc_path_curvature(ref_path)
    cur_curve = _calc_path_curvature(cur_path)
    distance, path = fastdtw(ref_curve, cur_curve, dist=2)
    return - np.tanh(0.5 * distance)


def incremental_reward(tar, delta):
    """ accu_reward on new pixels on target """
    return np.count_nonzero(delta + tar == 2) / np.multiply(*tar.shape)


def incremental_loss(tar, delta):
    """ loss on new pixels off target """
    return - np.count_nonzero(delta - tar == 1) / np.multiply(*tar.shape)


rewards_dict = {
    'img_cosine_reward': img_cosine_reward,
    'img_mse_loss': img_mse_loss,
    'scale_loss': scale_loss,
    'curvature_loss': curvature_loss,
    'iou_reward': iou_reward,
    'incremental_reward': incremental_reward,
    'incremental_loss': incremental_loss,
}


def traj_reward_fn(images, target):
    # increnemtal
    reward = 0
    for img0, img1 in zip(images, images[1:]):
        delta = img1 - img0
        instant_reward = incremental_reward(target, delta) - incremental_loss(target, delta)
        reward += instant_reward
    # final
    final_reward = img_cosine_reward(target, images[-1])
    reward += final_reward
    return reward
