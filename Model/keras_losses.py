import numpy as np
import tensorflow as tf


def dist_logits_loss(y_true, y_pred):
    mean, log_std = tf.split(y_pred, 2, axis=1)
    loss_mean = tf.reduce_mean(tf.losses.mse(y_true, mean))
    loss_std = tf.reduce_mean(tf.abs(log_std))  # log std应为0,

    return loss_mean + loss_std


def gaussian_loss(y_true, y_pred):
    mean, log_std = tf.split(y_pred, 2, axis=1)
    std = tf.exp(log_std)
    gaussian = mean + std * tf.random.normal(tf.shape(mean))
    loss = tf.reduce_mean(tf.math.square(gaussian - y_true))
    return loss


def make_direct_logits_loss_fn(nvec, use_z=False):
    nvec = nvec.tolist()
    if use_z:
        mask = np.array([True, True, False] * (len(nvec) // 3))

    def direct_logits_loss(y_true, y_pred):
        splited_logits = [logits for logits in tf.split(y_pred, nvec, axis=1)]
        labels = [y for y in tf.split(y_true, len(nvec), axis=1)]
        losses = [tf.losses.sparse_categorical_crossentropy(label, logits, from_logits=True) for
                  label, logits in zip(labels, splited_logits)]

        if not use_z:
            losses = tf.boolean_mask(losses, mask)

        loss = tf.reduce_mean(losses)
        return loss

    return direct_logits_loss
