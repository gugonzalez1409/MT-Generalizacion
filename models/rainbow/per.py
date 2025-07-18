import torch
import random
import numpy as np
from math import sqrt
from gym import spaces
from collections import deque
from typing import Any, Dict, List, Tuple, Union
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.type_aliases import ReplayBufferSamples



class PER(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        n_step: int = 3,
        gamma: float = 0.99,
        beta_initial: float = 0.45,
        beta_final: float = 1.0,
        beta_end_fraction: float = 1.0,
        optimize_memory_usage=False,
    ):
        super(ReplayBuffer, self).__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )

        self.beta_schedule = get_linear_fn(beta_initial, beta_final, beta_end_fraction)
        self.beta = beta_initial

        self.n_step = n_step
        self.gamma = gamma

        self.observations = np.full(self.buffer_size, None)
        self.next_observations = np.full(self.buffer_size, None)
        self.actions = torch.zeros((self.buffer_size, self.action_dim), dtype=torch.int64, device=self.device)
        self.rewards = torch.zeros((self.buffer_size,), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((self.buffer_size,), dtype=torch.float32, device=self.device)

        self.n_step_queue = [deque(maxlen=self.n_step + 1) for j in range(self.n_envs)]

        self.tree_priority_sum = np.full((2 * self.buffer_size), 0.0, dtype=np.float64)
        self.tree_priority_min = np.full((2 * self.buffer_size), np.inf, dtype=np.float64)

        self.max_priority = 1.0

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:

        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        action = action.reshape((self.n_envs, self.action_dim))

        for queue, obs, act, rwd, dns in zip(self.n_step_queue, obs, action, reward, done):

            queue.append((obs, act, rwd, dns))

            if len(queue) == self.n_step + 1 and not queue[0][3]:

                obs, act, rwd, _ = queue[0]
                next_obs, _, _, dns = queue[self.n_step]

                # n step reward
                for k in range(1, self.n_step):

                    rwd += queue[k][2] * self.gamma**k
                    if queue[k][3]:
                        dns = True
                        break

                self.observations[self.pos] = obs
                self.next_observations[self.pos] = next_obs
                self.actions[self.pos] = self.to_torch(act)
                self.rewards[self.pos] = self.to_torch(rwd)
                self.dones[self.pos] = self.to_torch(dns)

                self._set_priority_min(self.pos, sqrt(self.max_priority))
                self._set_priority_sum(self.pos, sqrt(self.max_priority))

                self.pos += 1
                if self.pos == self.buffer_size:
                    
                    self.full = True
                    self.pos = 0


    def _set_priority_min(self, i, priority_alpha):

        i += self.buffer_size
        self.tree_priority_min[i] = priority_alpha
        while i >= 2:
            i //= 2
            self.tree_priority_min[i] = min(
                self.tree_priority_min[2 * i], self.tree_priority_min[2 * i + 1]
            )

    def _set_priority_sum(self, i, priority):

        i += self.buffer_size
        self.tree_priority_sum[i] = priority
        while i >= 2:
            i //= 2
            self.tree_priority_sum[i] = (
                self.tree_priority_sum[2 * i] + self.tree_priority_sum[2 * i + 1]
            )

    def _sum(self):

        return self.tree_priority_sum[1]

    def _min(self):

        return self.tree_priority_min[1]

    def find_prefix_sum_idx(self, prefix_sum):

        i = 1
        while i < self.buffer_size:
            if self.tree_priority_sum[i * 2] > prefix_sum:
                i = 2 * i
            else:
                prefix_sum -= self.tree_priority_sum[i * 2]
                i = 2 * i + 1
        return i - self.buffer_size

    def update_priorities(self, indexes, priorities):

        for i, priority in zip(indexes, priorities):
            self.max_priority = max(self.max_priority, priority)
            priority_alpha = sqrt(priority)
            self._set_priority_min(i, priority_alpha)
            self._set_priority_sum(i, priority_alpha)

    def update_beta(self, current_progress_remaining: float):

        self.beta = self.beta_schedule(current_progress_remaining)

    def sample(self, batch_size: int, env: Union[VecNormalize, None] = None) -> Tuple[np.ndarray, torch.Tensor, ReplayBufferSamples]:

        weights = np.zeros(shape=batch_size, dtype=np.float32)
        indices = np.zeros(shape=batch_size, dtype=np.int32)

        for i in range(batch_size):

            p = random.random() * self._sum()
            index = self.find_prefix_sum_idx(p)
            indices[i] = index

        prob_min = self._min() / self._sum()
        max_weight = (prob_min * self.size()) ** (-self.beta)

        for i in range(batch_size):

            index = indices[i]
            prob = self.tree_priority_sum[index + self.buffer_size] / self._sum()
            weight = (prob * self.size()) ** (-self.beta)
            weights[i] = weight / max_weight

        return (indices, torch.from_numpy(weights).to(self.device), self._get_samples(indices, env))

    def obs_to_torch(self, array, copy=True):

        array = np.stack([np.array(obs, copy=False) for obs in array])
        return super().to_torch(array, copy)

    def _get_samples(self, batch_inds: np.ndarray, env: VecNormalize | None = None) -> ReplayBufferSamples:
        
        return ReplayBufferSamples(
            self.obs_to_torch(self.observations[batch_inds]),
            self.actions[batch_inds, :],
            self.obs_to_torch(self.next_observations[batch_inds]),
            self.dones[batch_inds].reshape(-1, 1),
            self.rewards[batch_inds].reshape(-1, 1),
        )

    def reset(self):
        
        super().reset()

        self.n_step_queue = [deque(maxlen=self.n_step + 1) for j in range(self.n_envs)]

        self.tree_priority_sum = np.full((2 * self.buffer_size), 0.0, dtype=np.float64)
        self.tree_priority_min = np.full((2 * self.buffer_size), np.inf, dtype=np.float64)

        self.max_priority = 1.0
