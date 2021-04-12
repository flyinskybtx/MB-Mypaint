import copy

import matplotlib.pyplot as plt
import numpy as np
import scipy
from skimage.draw import line

from Env.main_env import Canvas
from Model.action_embedder import ActionEmbedder
from Data.Deprecated.dynamics_model import CNPModel
from Data.Deprecated.repr_model import ReprModel
from utils.custom_rewards import traj_reward_fn


def make_reward_fn(target):
    def reward_fn(images):
        return traj_reward_fn(images, target)

    return reward_fn


def cem_optimize(init_mean, reward_func, init_variance=1., samples=20, precision=1e-2,
                 steps=5, nelite=5, constraint_mean=None, constraint_variance=(-999999, 999999)):
    """
        cem_optimize minimizes cost_function by iteratively sampling values around the current mean with a set variance.
        Of the sampled values the mean of the nelite number of samples with the lowest cost is the new mean for the next iteration.
        Convergence is met when either the change of the mean during the last iteration is less then precision.
        Or when the maximum number of steps was taken.
        :param init_mean: initial mean to sample new values around
        :param reward_func: varience used for sampling
        :param init_variance: initial variance
        :param samples: number of samples to take around the mean. Ratio of samples to elites is important.
        :param precision: if the change of mean after an iteration is less than precision convergence is met
        :param steps: number of steps
        :param nelite: number of best samples whose mean will be the mean for the next iteration
        :param constraint_mean: tuple with minimum and maximum mean
        :param constraint_variance: tuple with minumum and maximum variance
        :return: best_traj, best_reward
    """
    mean = init_mean.copy()
    variance = init_variance * np.ones_like(mean)

    step = 1
    diff = 999999
    best_traj = None
    best_reward = None

    while diff > precision and step < steps:
        # candidates: (horizon, samples, action_dim)
        candidates = np.stack(
            [np.random.multivariate_normal(m, np.diag(v), size=samples) for m, v in zip(mean, variance)])
        # apply actions constraints
        # process continuous random samples into available discrete context_actions
        candidates = np.clip(np.round(candidates), constraint_mean[0], constraint_mean[1]).astype(np.int)

        rewards = reward_func(candidates)
        print(rewards)
        sorted_idx = np.argsort(rewards)[::-1]  # descending
        best_reward = rewards[sorted_idx[0]]
        print(best_reward)
        best_traj = candidates[:, sorted_idx[0], :]
        print(best_traj)
        elite = candidates[:, sorted_idx[:nelite], :]  # elite: (horizon, nelite, action_dim)

        new_mean = np.mean(elite, axis=1)
        variance = np.var(elite, axis=1)
        diff = np.abs(np.mean(new_mean - mean))

        # update
        step += 1
        mean = new_mean

    return best_traj, best_reward  # select best to output


# TODO: Use XYZ rectify to modify real actions

class BaseMPCController:
    def __init__(self, config, action_space,
                 dynamics: CNPModel,
                 repr_model: ReprModel,
                 embedder: ActionEmbedder,
                 reward_fn, horizon=5, samples=10, **kwargs):
        """

        :param action_space:
        :param dynamics:
        :param reward_fn:
        :param horizon:
        :param samples:
        """
        self.config = config
        self.dynamics = dynamics
        self.repr_model = repr_model
        self.embedder = embedder
        self.horizon = horizon
        self.samples = samples
        self.action_space = action_space
        self.reward_fn = reward_fn

        self.state = None
        self.canvas = None
        self.xy = None
        self.z = None
        self.image_size = self.config.image_size
        self.action_disp = self.config.action_shape // 2
        self.z_grid = self.config.z_grid
        self.xy_grid = self.config.xy_grid

    def _rectify_trajs(self, trajs):
        """ Clipping trajs where xs, ys, zs is outof bound

        :param trajs: np.ndarray of shape (Horizon, Sample, Action_shape)
        :return:
        """
        real_trajs = []
        horizon, sample, action_shape = trajs.shape
        for i in range(sample):
            xyz = np.array([self.xy[0], self.xy[1], self.z])
            real_traj = []
            for action in trajs[:, i, :]:
                delta = self._from_action(action)
                new_xyz = np.clip(xyz + delta,
                                  np.array([0, 0, 0]),
                                  np.array([self.image_size, self.image_size, 1]))
                real_delta = new_xyz - xyz
                real_action = self._to_action(real_delta)
                real_traj.append(real_action)
                xyz = new_xyz
                if xyz[2] == 0:  # IF zs is 0, consider finished
                    break
            real_traj = np.stack(real_traj +
                                 [np.array([self.action_disp] * 3) for _ in
                                  range(horizon - len(real_traj))], axis=0)
            real_trajs.append(real_traj)
        return np.stack(real_trajs, axis=1)

    def _from_action(self, action):
        dx = (action[0] - self.action_disp) * self.xy_grid / self.action_disp
        dy = (action[1] - self.action_disp) * self.xy_grid / self.action_disp
        dz = (action[2] - self.action_disp) * self.z_grid
        return np.array([dx, dy, dz])

    def _to_action(self, arr):
        xyz = arr / np.array([self.xy_grid, self.xy_grid, self.z_grid]) * \
              np.array([self.action_disp, self.action_disp, 1]) + \
              np.array([self.action_disp, self.action_disp, self.action_disp])

        xyz = np.round(xyz).astype(int)
        return xyz

    def _expected_reward(self, trajs):
        # trajs shape: (horizon, num_samples, action_dims)
        num_samples = trajs.shape[1]
        states = np.repeat(np.expand_dims(self.state, axis=0), num_samples, axis=0)  # shape (num_samples, state_dims)
        latents = self.repr_model.latent_encode(states)
        deltas_history = self.calculate_deltas(latents, trajs)
        images_history = self._calculate_images(deltas_history, trajs)

        self.images_history = copy.deepcopy(images_history)

        rewards = [self.reward_fn(images) for images in images_history]
        return np.array(rewards)

    def calculate_deltas(self, latents, trajs):
        history = []
        for actions in trajs:
            embedding = self.embedder.transform(actions)
            query_x = np.concatenate([latents, embedding], axis=-1)
            target_y = self.dynamics.predict(query_x)
            latents += target_y['mu']
            deltas = self.repr_model.latent_decode(latents).squeeze(axis=-1)
            history.append(deltas.copy())
        return history

    def _calculate_images(self, history, trajs):
        canvases = [copy.deepcopy(self.canvas) for _ in range(self.samples)]
        positions = [copy.deepcopy(self.xy) for _ in range(self.samples)]
        zs = [copy.deepcopy(self.z) for _ in range(self.samples)]
        dones = [False for _ in range(self.samples)]

        history_images = [[] for _ in range(self.samples)]
        for deltas, actions in zip(history, trajs):
            for i, canvas in enumerate(canvases):
                if np.all(dones):
                    return history_images
                if dones[i]:
                    continue
                dx, dy, dz = self._from_action(actions[i])
                positions[i] += np.array([dx, dy])
                positions[i] = np.clip(positions[i],
                                       0,
                                       self.config.image_size - 1)
                zs[i] += dz
                zs[i] = np.clip(zs[i], 0, 1)
                if zs[i] == 0:
                    dones[i] = True

                canvas.place_delta(deltas[i], positions[i])
                history_images[i].append(canvas.frames)
        return history_images

    def preset(self, canvas: Canvas, xy):
        self.canvas = canvas
        self.xy = np.array(xy)

    def _display_prediction(self, traj, image):
        x, y = self.xy
        z = self.z
        action_disp = self.config.action_shape // 2
        xy_grid = self.config.xy_grid
        z_grid = self.config.z_grid
        waypoints_frame = np.zeros_like(self.canvas.frames)
        conv_kernel = np.ones(shape=(5, 5))

        for action in traj:
            local_frame = np.zeros_like(self.canvas.frames)
            new_x = x + (action[0] - action_disp) * xy_grid / action_disp
            new_y = y + (action[1] - action_disp) * xy_grid / action_disp
            new_z = z + (action[2] - action_disp) * z_grid / action_disp
            local_frame[int(new_x), int(new_y)] = float(new_z)
            local_frame = scipy.ndimage.convolve(local_frame, conv_kernel, mode='constant', cval=0.0)
            rr, cc = line(int(x), int(y), int(new_x), int(new_y))
            local_frame[rr, cc] = new_z
            waypoints_frame += scipy.ndimage.convolve(local_frame, conv_kernel, mode='constant', cval=0.0)
            x, y, z = new_x, new_y, new_z
            waypoints_frame = np.clip(waypoints_frame, 0, 1)
        return np.stack([image, self.canvas.frames, waypoints_frame], axis=-1)


class ShootingMPCController(BaseMPCController):
    def __init__(self, config, action_space,
                 dynamics: CNPModel,
                 repr_model: ReprModel,
                 embedder: ActionEmbedder,
                 reward_fn, horizon=5, samples=10, **kwargs):
        super().__init__(config, action_space, dynamics, repr_model, embedder, reward_fn, horizon, samples, **kwargs)
        self.name = 'Naive_MPC'

    def next_action(self, obs, print_expectation=False):
        """

        :param print_expectation:
        :param obs:
        :return: single actions
        """
        # Random Shooting
        self.state = obs
        self.z = obs[0, 0, 3]
        # --- Initialize candidat tracks
        trajs = []
        for _ in range(self.horizon):
            trajs.append(np.array([self.action_space.sample() for _ in range(self.samples)]).reshape(self.samples, -1))
        trajs = np.stack(trajs, axis=0)  # Horizon * Sample * Action_shape
        # --- ---
        trajs = self._rectify_trajs(trajs)  # Horizon * Sample * Action_shape

        rewards = self._expected_reward(trajs)
        sorted_idx = np.argsort(rewards)[::-1]  # descending
        best_idx = sorted_idx[0]
        best_traj = trajs[:, best_idx, :]

        if print_expectation:
            print('Best reward_names:', rewards[sorted_idx])  # todo
            print('Best trajectory:', best_traj)
            best_image = self.images_history[best_idx][-1]
            result_image = self._display_prediction(best_traj, best_image)

            plt.imshow(result_image)
            plt.suptitle(f'MPC expected result, reward={rewards[sorted_idx[0]]}')
            plt.show()

        return best_traj[0]


class CemMPCController(BaseMPCController):
    def __init__(self, config, action_space, dynamics: CNPModel, repr_model: ReprModel, embedder: ActionEmbedder,
                 reward_fn, horizon=5, samples=10, strides=1, **kwargs):
        """

        :param action_space:
        :param dynamics:
        :param reward_fn:
        :param horizon:
        :param samples:
        """
        super().__init__(config, action_space, dynamics, repr_model, embedder, reward_fn, horizon, samples, **kwargs)
        self.name = 'CemMPCController'
        self.strides = kwargs.setdefault('strides', 1)

        self.trajectory = None
        self.buffer = []

    def next_action(self, obs, print_expectation=False):
        """

        :param print_expectation:
        :param obs:
        :return: single actions
        """
        if len(self.buffer) > 0:
            return self.buffer.pop(0)
        else:
            self.update_buffer(obs, print_expectation)
            return self.buffer.pop(0)

    def update_buffer(self, obs, print_expectation=False):
        self.state = obs
        self.z = obs[0, 0, 3]

        if self.trajectory is None:
            # Randomly initialize trajectory
            self.trajectory = np.concatenate(
                [np.array([self.action_space.sample()]).reshape(1, -1) for _ in range(self.horizon)])

        self.trajectory, self.expectation = cem_optimize(self.trajectory,
                                                         self._expected_reward,
                                                         constraint_mean=[0, self.config.action_shape - 1],
                                                         samples=self.samples)

        # update trajectory for next step
        self.buffer = list(self.trajectory[:self.strides])
        self.trajectory = np.concatenate([
            self.trajectory[self.strides:],
            np.array([self.action_space.sample() for _ in range(self.strides)]).reshape(self.strides, -1)
        ])
        if print_expectation:
            print('Reward expectation: ', self.expectation)
