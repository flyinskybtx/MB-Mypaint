import numpy as np

from Env.canvas import CanvasFactory, Canvas


class Planner:
    def __init__(self, action_space, reward_fn, dynamics, horizon=5, num_particles=50):
        self.action_space = action_space
        self.dynamics = dynamics
        self.reward_fn = reward_fn
        self.horizon = horizon
        self.num_particles = num_particles

        self.canvas = None

    def gen_trajs(self):
        trajs = np.apply_along_axis(lambda x: self.action_space.sample(),
                                    -1, np.zeros((self.num_particles, self.horizon, 1)))
        return trajs

    def plan(self):
        canvas_factory = CanvasFactory(self.canvas, self.num_particles, self.dynamics)
        self.dynamics.initialize(canvas_factory.deltas)

        trajs = self.gen_trajs()
        for i in range(self.horizon):
            actions = trajs[:, i, :]
            canvas_factory.step(actions)
        frames = canvas_factory.frames
        rewards = [np.sum(list(self.reward_fn(frame).values())) for frame in frames]
        sorted_idx = np.argsort(rewards)[::-1]  # descending
        best_idx = sorted_idx[0]
        best_traj = trajs[best_idx, :, :]
        best_reward = rewards[best_idx]
        return best_traj, best_reward

    def update(self, canvas):
        self.canvas = canvas
