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

        # scheduler para beta
        self.beta_schedule = get_linear_fn(beta_initial, beta_final, beta_end_fraction)
        self.beta = beta_initial

        # valor de n_step learning y factor de descuento
        self.n_step = n_step
        self.gamma = gamma

        # arreglos que contienen los valores de las transiciones
        self.observations = np.full(self.buffer_size, None)
        self.next_observations = np.full(self.buffer_size, None)
        self.actions = torch.zeros((self.buffer_size, self.action_dim), dtype=torch.int64, device=self.device)
        self.rewards = torch.zeros((self.buffer_size,), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((self.buffer_size,), dtype=torch.float32, device=self.device)

        # guarda los valores de las transiciones n-step
        self.n_step_buffers = [deque(maxlen=self.n_step + 1) for j in range(self.n_envs)]

        # arreglos para las prioridades maximas y minimas
        self.priority_sum = np.full((2 * self.buffer_size), 0.0, dtype=np.float64)
        self.priority_min = np.full((2 * self.buffer_size), np.inf, dtype=np.float64)

        # prioridad inicial
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
        
        # SB3
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        action = action.reshape((self.n_envs, self.action_dim))

        for queue, o, a, r, d in zip(self.n_step_buffers, obs, action, reward, done):
            queue.append((o, a, r, d))

            if len(queue) == self.n_step + 1 and not queue[0][3]:
                o, a, r, _ = queue[0]
                no, _, _, d = queue[self.n_step]

                for k in range(1, self.n_step):
                    r += queue[k][2] * self.gamma**k
                    if queue[k][3]:
                        d = True
                        break

                self.observations[self.pos] = o
                self.next_observations[self.pos] = no

                self.actions[self.pos] = self.to_torch(a)
                self.rewards[self.pos] = self.to_torch(r)
                self.dones[self.pos] = self.to_torch(d)

                # dar prioridades iniciales a las transiciones
                self._set_priority_min(self.pos, sqrt(self.max_priority))
                self._set_priority_sum(self.pos, sqrt(self.max_priority))

                self.pos += 1
                if self.pos == self.buffer_size:
                    self.full = True
                    self.pos = 0

    def _set_priority_min(self, idx, priority_alpha):

        idx += self.buffer_size
        self.priority_min[idx] = priority_alpha
        while idx >= 2:
            idx //= 2
            self.priority_min[idx] = min(
                self.priority_min[2 * idx], self.priority_min[2 * idx + 1]
            )

    def _set_priority_sum(self, idx, priority):

        idx += self.buffer_size
        self.priority_sum[idx] = priority
        while idx >= 2:
            idx //= 2
            self.priority_sum[idx] = (
                self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]
            )

    def _sum(self):

        return self.priority_sum[1]

    def _min(self):
        
        return self.priority_min[1]

    def find_prefix_sum_idx(self, prefix_sum):

        idx = 1
        while idx < self.buffer_size:
            if self.priority_sum[idx * 2] > prefix_sum:
                idx = 2 * idx
            else:
                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2 * idx + 1
        return idx - self.buffer_size

    def update_priorities(self, indexes, priorities):

        for idx, priority in zip(indexes, priorities):
            self.max_priority = max(self.max_priority, priority)
            priority_alpha = sqrt(priority)
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)

    def update_beta(self, current_progress_remaining: float):
        
        self.beta = self.beta_schedule(current_progress_remaining)

    def sample(self, batch_size: int, env: Union[VecNormalize, None] = None) -> Tuple[np.ndarray, torch.Tensor, ReplayBufferSamples]:
        
        weights = np.zeros(shape=batch_size, dtype=np.float32)
        indices = np.zeros(shape=batch_size, dtype=np.int32)

        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            indices[i] = idx

        prob_min = self._min() / self._sum()
        max_weight = (prob_min * self.size()) ** (-self.beta)

        for i in range(batch_size):
            
            idx = indices[i]
            prob = self.priority_sum[idx + self.buffer_size] / self._sum()
            weight = (prob * self.size()) ** (-self.beta)
            weights[i] = weight / max_weight

        return (indices, torch.from_numpy(weights).to(self.device), self._get_samples(indices, env),)

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

        self.n_step_buffers = [deque(maxlen=self.n_step + 1) for j in range(self.n_envs)]

        self.priority_sum = np.full((2 * self.buffer_size), 0.0, dtype=np.float64)
        self.priority_min = np.full((2 * self.buffer_size), np.inf, dtype=np.float64)

        self.max_priority = 1.0