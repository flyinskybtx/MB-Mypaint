import numpy as np

from Env.canvas import CanvasFactory, Canvas
import matplotlib.pyplot as plt

class CemPlanner:
    def __init__(self, action_space, reward_fn, dynamics, horizon=5, num_particles=50):
        self.action_space = action_space
        self.dynamics = dynamics
        self.reward_fn = lambda x: np.sum(list(reward_fn(x).values()))
        self.horizon = horizon
        self.num_particles = num_particles

        self.canvas = None
        self.traj = None

    def update_canvas(self, canvas):
        self.canvas = canvas

    def random_traj(self, horizon):
        return np.stack([self.action_space.sample() for _ in range(horizon)], axis=0)

    def initialize_traj(self):
        if self.traj is None:
            self.traj = self.random_traj(self.horizon)
        else:
            cur_len = self.traj.shape[0]
            if cur_len < self.horizon:
                self.traj = np.concatenate([self.traj, self.random_traj(self.horizon - cur_len)], axis=0)

    @staticmethod
    def sample_traj(mean, var):
        m = mean.reshape(-1, )
        v = var.reshape(-1, )
        traj = np.random.multivariate_normal(m, np.diag(v)).reshape(-1, 3)
        return traj

    def predict_frame(self, trajs):
        canvas_factory = CanvasFactory(self.canvas, self.num_particles, self.dynamics)
        deltas = np.repeat((np.expand_dims(self.canvas.delta, axis=0)), repeats=self.num_particles, axis=0)
        self.dynamics.initialize(deltas)

        for i in range(trajs.shape[1]):
            actions = trajs[:, i, :]
            canvas_factory.step(actions)
        frames = canvas_factory.frames
        frames = np.clip(frames, 0, 1)
        return frames

    def optimize(self, variance=2., precision=1e-2, steps=5, nelite=5, action_constraint=(0, 4)):
        mean = self.traj.reshape(-1, )
        var = variance * np.ones_like(mean)

        step = 1
        diff = 999999

        while diff > precision and step < steps:
            # Candidates: (horizon, samples, action_dim)
            candidates = np.stack([self.sample_traj(mean, var) for _ in range(self.num_particles)], axis=0)
            # Apply actions constraints
            candidates = np.clip(np.round(candidates), action_constraint[0], action_constraint[1]).astype(np.int)
            # Predict result image
            frames = self.predict_frame(candidates)
            # Rewards
            rewards = list(map(self.reward_fn, frames))

            sorted_idx = np.argsort(rewards)[::-1]  # descending
            best_idx = sorted_idx[0]
            best_traj = candidates[best_idx, :, :]
            best_reward = rewards[best_idx]
            elite = candidates[sorted_idx[:nelite], :, :]  # elite: (horizon, nelite, action_dim)
            new_mean = np.mean(elite, axis=0)
            new_mean = new_mean.reshape(-1, )
            var = np.var(elite, axis=0)
            var = var.reshape(-1, )
            diff = np.abs(np.mean(new_mean - mean))

            print(best_traj)
            print(best_reward)
            plt.imshow(frames[best_idx])
            plt.show()

            # update
            step += 1
            mean = new_mean

        return best_traj, best_reward

    def plan(self):
        self.initialize_traj()
        best_traj, best_reward = self.optimize()
        self.traj = best_traj[1:]
        action = best_traj[0].tolist()
        return action, best_reward
