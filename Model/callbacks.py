import scipy
import os
import random

import numpy
from matplotlib import pyplot as plt
from skimage import transform
from skimage.draw import draw

from tensorflow import keras
import tensorflow as tf
import numpy as np

from Data import obs_to_delta


class GradExtendedTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, val_data, **kwargs):
        super().__init__(**kwargs)
        self.val_data = val_data

    def _log_gradients(self, epoch):
        writer = self._train_writer

        with writer.as_default(), tf.GradientTape() as g:
            # here we use test data_loader to calculate the gradients
            X, Y = self.val_data

            y_pred = self.model(X)  # forward-propagation
            loss = self.model.compiled_loss(y_true=tf.convert_to_tensor(Y), y_pred=y_pred)  # calculate loss
            gradients = g.gradient(loss, self.model.trainable_weights)  # back-propagation

            # In eager mode, grads does not have name, so we get names from model.trainable_weights
            for weights, grads in zip(self.model.trainable_weights, gradients):
                tf.summary.histogram(
                    weights.name.replace(':', '_') + '_grads', data=grads, step=epoch)

        writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        # This function overwrites the on_epoch_end in tf.keras.callbacks.TensorBoard
        # but we do need to run the original on_epoch_end, so here we use the super function.
        super(GradExtendedTensorBoard, self).on_epoch_end(epoch, logs=logs)

        if self.histogram_freq and epoch % self.histogram_freq == 0:
            self._log_gradients(epoch)


class VaeVisCallback(keras.callbacks.Callback):
    def __init__(self, data, frequency, img_dir=None, total_count=10):
        super().__init__()
        self.data = data
        self.frequency = frequency
        self.total_count = total_count
        self.img_dir = img_dir

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.frequency != 0:
            return
        X = numpy.copy(self.data)
        numpy.random.shuffle(X)
        Y_ = self.model(X)

        if self.img_dir is not None:
            os.makedirs(self.img_dir, exist_ok=True)
        count = 0
        for y_pred, y_true in zip(Y_, X):
            if numpy.max(y_true) > 0:
                frame = numpy.concatenate([y_pred, y_true], axis=1)  # concat along ys axis for view
                fig = plt.figure(f'Epoch:{epoch}')
                error = numpy.mean((Y_ - X) ** 2)
                fig.suptitle(f'MSE: {error}')
                plt.imshow(frame)
                plt.show()
                if self.img_dir is not None:
                    fig.savefig(os.path.join(self.img_dir, f'epoch{epoch}_{count + 1}.png'), format='png')

                plt.close(fig)
                count += 1
            if count >= self.total_count:
                break


class CnpVisCallback(keras.callbacks.Callback):
    def __init__(self, data, obs_encoder, obs_decoder, embedder, num_context, frequency=5, total=10):
        super().__init__()
        self.data = data.data
        self.shuffle = data.shuffle
        self.encoder = obs_encoder
        self.decoder = obs_decoder
        self.embedder = embedder
        self.num_context = num_context
        self.frequency = frequency
        self.total = total

    def _get_epoch_data(self):
        self.shuffle(self.data)
        key = random.choice(list(self.data.keys()))
        data = self.data[key]

        context_obs = obs_to_delta(data['obs'][:self.num_context])
        context_new_obs = obs_to_delta(data['new_obs'][:self.num_context])
        context_actions = data['actions'][:self.num_context]

        query_obs = obs_to_delta(data['obs'][self.num_context:])
        query_new_obs = obs_to_delta(data['new_obs'][self.num_context:])
        query_actions = data['actions'][self.num_context:]
        num_test = data['actions'].shape[0] - self.num_context

        context_x_0 = self.encoder(context_obs)
        context_x_1 = self.encoder(context_new_obs)
        if self.embedder is not None:
            context_u = self.embedder(context_actions)
        else:
            context_u = context_actions
        context_x = np.concatenate([context_x_0, context_u], axis=-1)  # [latent, actions]
        context_y = context_x_1 - context_x_0  # latent1 - latent0
        context_x = np.repeat(np.expand_dims(context_x, axis=0), repeats=num_test, axis=0)
        context_y = np.repeat(np.expand_dims(context_y, axis=0), repeats=num_test, axis=0)

        query_x_0 = self.encoder(query_obs)
        if self.embedder is not None:
            query_u = self.embedder(query_actions)
        else:
            query_u = query_actions
        query_x = np.concatenate([query_x_0, query_u], axis=-1)
        y_1 = self.encoder(query_new_obs)

        return (context_x, context_y, query_x), (query_x_0, y_1), (query_obs, query_new_obs)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.frequency != 0:
            return

        query, (y0, y), (Y0, Y_true) = self._get_epoch_data()
        Y0 = Y0.squeeze(axis=-1)
        Y_true = Y_true.squeeze(axis=-1)

        dy = self.model(query)
        y_hat = y0 + dy

        mse_error = np.round(np.mean(np.square(y_hat - y), axis=0), decimals=2)
        print(f"MSE on all dim: {np.mean(mse_error)}, \nMSE on each dim: {mse_error}")

        recon = self.decoder(y).numpy().squeeze(axis=-1)
        recon_hat = self.decoder(y_hat).numpy().squeeze(axis=-1)

        mse_error = np.mean(np.square(recon_hat - recon))
        print(f"MSE on recon image = {mse_error}")

        count = 0
        for y_pred, y_true, y_prev, y_gt in zip(recon_hat, recon, Y0, Y_true):
            if count >= self.total:
                break
            if np.max(y_true) > 0:
                fig = plt.figure(f'Epoch:{epoch}')
                error = np.mean((y_pred - y_true) ** 2)
                fig.suptitle(f'MSE: {error}')

                ax1 = fig.add_subplot(221)
                ax1.title.set_text("y_pred")
                ax1.imshow(y_pred)
                ax1 = fig.add_subplot(222)
                ax1.title.set_text("y_true")
                ax1.imshow(y_true)
                ax1 = fig.add_subplot(223)
                ax1.title.set_text("y_prev")
                ax1.imshow(y_prev)
                ax1 = fig.add_subplot(224)
                ax1.title.set_text("y_gt")
                ax1.imshow(y_gt)

                plt.show()
                plt.close(fig)

                count += 1


class MlpVisCallback(keras.callbacks.Callback):
    def __init__(self, data, obs_encoder, obs_decoder, embedder, frequency=5, total=10):
        super().__init__()
        self.data = data.data
        self.shuffle = data.shuffle
        self.encoder = obs_encoder
        self.decoder = obs_decoder
        self.embedder = embedder
        self.frequency = frequency
        self.total = total

    def _get_epoch_data(self):
        self.shuffle(self.data)
        key = random.choice(list(self.data.keys()))
        data = self.data[key]

        obs = obs_to_delta(data['obs'])
        new_obs = obs_to_delta(data['new_obs'])
        actions = data['actions']

        x_0 = self.encoder(obs)
        u = self.embedder(actions)
        xu = np.concatenate([x_0, u], axis=-1)
        x_1 = self.encoder(new_obs)
        z = data['obs'][:, 0, 0, 3].reshape(-1, 1)

        return (xu, z), (x_0, x_1), (obs, new_obs)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.frequency != 0:
            return

        (xu, z), (x_0, x_1), (Y0, Y_true) = self._get_epoch_data()
        Y0 = Y0.squeeze(axis=-1)
        Y_true = Y_true.squeeze(axis=-1)

        dx = self.model([xu, z])
        x_hat = x_0 + dx

        mse_error = np.round(np.mean(np.square(x_hat - x_1), axis=0), decimals=2)
        print(f"MSE on all dim: {np.mean(mse_error)}, \nMSE on each dim: {mse_error}")

        recon = self.decoder(x_1).numpy().squeeze(axis=-1)
        recon_hat = self.decoder(x_hat).numpy().squeeze(axis=-1)

        mse_error = np.mean(np.square(recon_hat - recon))
        print(f"MSE on recon image = {mse_error}")

        count = 0
        for y_pred, y_true, y_prev, y_gt in zip(recon_hat, recon, Y0, Y_true):
            if count >= self.total:
                break
            if np.max(y_true) > 0:
                fig = plt.figure(f'Epoch:{epoch}')
                error = np.mean((y_pred - y_true) ** 2)
                fig.suptitle(f'MSE: {error}')

                ax1 = fig.add_subplot(221)
                ax1.title.set_text("y_pred")
                ax1.imshow(y_pred)
                ax1 = fig.add_subplot(222)
                ax1.title.set_text("y_true")
                ax1.imshow(y_true)
                ax1 = fig.add_subplot(223)
                ax1.title.set_text("y_prev")
                ax1.imshow(y_prev)
                ax1 = fig.add_subplot(224)
                ax1.title.set_text("y_gt")
                ax1.imshow(y_gt)

                plt.show()
                plt.close(fig)

                count += 1


class AnnealingCallback(keras.callbacks.Callback):
    def __init__(self, weight, start=20, anneal_time=40):
        super().__init__()
        self.start = start
        self.anneal_time = anneal_time
        self.weight = weight

    def on_epoch_begin(self, epoch, logs=None):
        if epoch >= self.start:
            if epoch <= self.start + self.anneal_time:
                new_weight = min(keras.backend.get_value(self.weight) + (1. / self.anneal_time), 1.)
            else:
                new_weight = 1
            keras.backend.set_value(self.weight, new_weight)
        print("Current Anneal Weight is " + str(keras.backend.get_value(self.weight)))


class RobotDirectEnvVisCallback(keras.callbacks.Callback):
    def __init__(self, data_loader, nvec, num_xy, num_z, image_size, frequency=5, total=10, img_dir=None):
        super().__init__()
        self.data_loader = data_loader
        self.shuffle = data_loader.shuffle
        self.frequency = frequency
        self.total = total
        self.img_dir = img_dir
        self.nvec = nvec
        self.num_xy = num_xy
        self.num_z = num_z
        self.image_size = image_size

    def action_to_frame(self, action):
        action = action.reshape(-1, 3)
        waypoints = (action + np.array([0.5, 0.5, 0])) / \
                    np.array([self.num_xy, self.num_xy, self.num_z]) * \
                    np.array([self.image_size, self.image_size, 1])

        wps_frame = np.zeros((self.image_size, self.image_size))

        for wp in waypoints:
            wps_frame[int(wp[0]), int(wp[1])] = wp[2]
        kernel = np.ones((int(self.image_size / self.num_xy), int(self.image_size / self.num_xy)))
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


class ContinuousEnvVisCallback(keras.callbacks.Callback):
    def __init__(self, data_loader, frequency=5, total=10):
        super().__init__()
        self.data_loader = data_loader
        self.frequency = frequency
        self.total = total

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.frequency != 0:
            return

        i = np.random.randint(0, self.data_loader.__len__())
        obs, (action_true, _) = self.data_loader.__getitem__(i)
        logits, _ = self.model(obs)
        logits = logits.numpy()

        for img, logit, y_action in zip(obs[:self.total], logits[:self.total], action_true[:self.total]):
            splited = np.split(logit, 2, axis=-1)  # split return N+1 values
            action = splited[0]
            log_var = splited[1]
            var = np.exp(log_var)
            obs_size = img.shape[0]
            start = (int(obs_size / 2), int(obs_size / 2))
            end = (int((obs_size-1) * action[0] / 2 + obs_size/2),
                   int((obs_size-1) * action[1] / 2 + obs_size/2))

            rr, cc, val = draw.line_aa(*start, *end)
            frame_pred = np.zeros((obs_size, obs_size))
            frame_pred[rr, cc] = val * 0.5 + action[2] * 0.5

            end = (int((obs_size-1) * y_action[0] / 2 + obs_size/2),
                   int((obs_size-1) * y_action[1] / 2 + obs_size/2))
            rr, cc, val = draw.line_aa(*start, *end)
            frame_true = np.zeros((obs_size, obs_size))
            frame_true[rr, cc] = val * 0.5 + y_action[2] * 0.5
            frame = np.stack([frame_pred, np.zeros_like(frame_pred), frame_true], axis=-1)

            fig = plt.figure()
            fig.suptitle(f'Action: {action} (+- {var})/ {y_action}')
            ax = plt.subplot(121)
            plt.imshow(img[:, :, :3])
            ax.set_title("Obs")
            ax = plt.subplot(122)
            plt.imshow(frame)
            ax.set_title("Action")
            plt.show()


class AeVisCallback(keras.callbacks.Callback):
    def __init__(self, data, frequency, include_z=False, img_dir=None, total_count=10):
        super().__init__()
        self.data = data
        self.frequency = frequency
        self.total_count = total_count
        self.img_dir = img_dir
        self.include_z = include_z

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.frequency != 0:
            return
        X = np.copy(self.data)
        np.random.shuffle(X)
        Y_ = self.model(X)

        count = 0
        for y_pred, y_true in zip(Y_, X):
            frame = np.concatenate([y_pred[:, :, 0], y_true[:, :, 0]], axis=1)  # concat along ys axis for view
            fig = plt.figure(f'Epoch:{epoch}')
            if self.include_z:
                fig.suptitle(f'Z: {np.mean(y_pred[:, :, 1])} - {np.mean(y_true[:, :, 1])}')
            plt.imshow(frame)
            plt.show()
            plt.close(fig)
            count += 1
            if count >= self.total_count:
                break
