import copy

import numpy as np

from Env.main_env import Canvas
from Model.dynamics_model import ActionEmbedder, CNPModel
from Model.repr_model import ReprModel
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

    while diff > precision and step < steps:
        # candidates: (horizon, samples, action_dim)
        candidates = np.stack(
            [np.random.multivariate_normal(m, np.diag(v), size=samples) for m, v in zip(mean, variance)])
        # apply action constraints
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
        self.action_disp = self.config.action_shape // 2
        self.z_grid = self.config.z_grid
        self.xy_grid = self.config.xy_grid

    def _expected_reward(self, trajs):
        # trajs shape: (horizon, num_samples, action_dims)
        num_samples = trajs.shape[1]
        states = np.repeat(np.expand_dims(self.state, axis=0), num_samples, axis=0)  # shape (num_samples, state_dims)
        latents = self.repr_model.latent_encode(states)
        latents_history = self._calculate_latents(latents, trajs)
        images_histroy = self._calculate_images(latents_history, trajs)

        rewards = [self.reward_fn(images) for images in images_histroy]
        return np.array(rewards)

    def _calculate_latents(self, latents, trajs):
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
                dx, dy, dz = self._calculate_disp(actions[i])
                positions[i] += np.array([dx, dy])
                positions[i] = np.clip(positions[i],
                                       0,
                                       self.config.image_size - 1)
                zs[i] += dz
                zs[i] = np.clip(zs[i], 0, 1)
                if zs[i] == 0:
                    dones[i] = True

                canvas.place_delta(deltas[i], positions[i])
                history_images[i].append(canvas.frame)
        return history_images

    def _calculate_disp(self, action):
        dx = (action[0] - self.action_disp) * self.xy_grid / self.action_disp
        dy = (action[1] - self.action_disp) * self.xy_grid / self.action_disp
        dz = (action[2] - self.action_disp) * self.z_grid
        return np.array([dx, dy, dz])

    def preset(self, canvas: Canvas, xy):
        self.canvas = canvas
        self.xy = np.array(xy)


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
        :return: single action
        """
        # Random Shooting
        self.state = obs
        self.z = obs[0, 0, 3]
        trajs = []
        for _ in range(self.horizon):
            trajs.append(np.array([self.action_space.sample() for _ in range(self.samples)]).reshape(self.samples, -1))
        trajs = np.stack(trajs, axis=0)

        rewards = self._expected_reward(trajs)
        print(rewards)  # todo
        sorted_idx = np.argsort(rewards)[::-1]  # descending
        best_reward = rewards[sorted_idx[0]]
        print(best_reward)  # todo
        best_traj = trajs[:, sorted_idx[0], :]
        print(best_traj)  # todo

        if print_expectation:
            print('Reward expectation: ', best_reward)
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
        :return: single action
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
