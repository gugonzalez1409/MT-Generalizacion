import numpy as np
import torch as th
from math import sqrt
from gym import spaces
from collections import deque
from typing import Any, Dict, List, Union, Optional
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize

"""
Prioritized Experience Replay buffer con n-step returns.

"""

# algunas partes del codigo son sacadas de la clase ReplayBuffer de SB3

class PER(ReplayBuffer):

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "cpu",
            n_envs: int = 1,
            n_steps: int= 3,
            gamma: float = 0.99,
            optimize_memory_usage = False,
            handle_timeout_termination = True,
        ):
        
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination)
        self.n_steps = n_steps
        self.gamma = gamma
        
        self.n_steps_buffer = [deque(maxlen=self.n_steps + 1) for _ in range(self.n_envs)]
        # almacenar la suma de las prioridades para cada elemento
        self.sum_prior = np.full((2 * self.buffer_size), 0., dtype=np.float32)
        # guardar el minimo de las prioridades
        self.min_prior = np.full((2 * self.buffer_size), np.inf, dtype=np.float32)
        self.max_prior = 1.0


    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
        ) -> None:

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)
            next_obs = next_obs.reshape((self.n_envs,) + self.obs_shape)

        # Same, for actions
        if isinstance(self.action_space, spaces.Discrete):
            action = action.reshape((self.n_envs, self.action_dim))

        for queue, o, a, r, d in zip(self.n_steps_buffer, obs, action, reward, done):
            queue.append((o, a, r, d))

            # si la cola está llena y el primer elemento no tiene done = true, calcula el n-steps return
            if (len(queue) == self.n_steps + 1 and not queue[0][3]):
                o, a, r, _ = queue[0]
                no, _, _, d = queue[self.n_steps]

                # n steps return
                for i in range (1, self.n_steps):
                    r += queue[i][2] * self.gamma ** i
                    if queue[i][3]:
                        d = True
                        break
                
                # agrega al buffer
                super().add(
                    obs = o,
                    next_obs = no,
                    action = a,
                    reward = r,
                    dones = d,
                    infos = infos
                )
                ## setear prioridad
                self.set_min_prior(self.pos, sqrt(self.max_prior))
                self.set_sum_prior(self.pos, sqrt(self.max_prior))

    def set_min_prior(self, i, alpha):

        # setear la prioridad minima para el elemento i
        i += self.buffer_size
        self.min_prior[i] = alpha

        while i >= 2:
            i = i // 2
            self.min_prior[i] = min(self.min_prior[2 * i], self.min_prior[2 * i + 1 ]) # minimo entre los hijos

    def set_sum_prior(self, i, p):

        # setear la prioridad para el elemento i
        i += self.buffer_size
        self.sum_prior[i] = p

        while i >= 2:
            i = i // 2
            self.sum_prior[i] = self.sum_prior[2 * i] + self.sum_prior[2 * i + 1] # suma de los hijos

    
    def find_idx(self, sum):
        # encontrar el indice del elemento que tiene la suma mayor o igual a sum

        i = 1
        while i < self.buffer_size:
            if self.sum_prior[2 * i] >= sum:
                i = 2 * i
            else:
                sum -= self.sum_prior[2 * i]
                i = 2 * i + 1

        return i - self.buffer_size # restar el tamaño del buffer para obtener el indice real
    

    def update_priorities(self, i, priors):
        # actualizar las prioridades
        for i, prior in zip(i, priors):
            self.max_prior[i] = max(self.max_prior, prior)
            alpha = sqrt(prior)
            self.set_min_prior(i, alpha)
            self.set_sum_prior(i, alpha)


    def sample(self, batch_size: int, beta: float, env: Union[VecNormalize, None] = None):

        weights = np.zeros((batch_size,), dtype=np.float32)
        indices = np.zeros((batch_size,), dtype=np.int64)

        for i in range(batch_size):
            p = np.random.random() * self.get_sum()
            idx = self.find_idx(p)
            indices[i] = idx

        min_prob = self.get_min() / self.get_sum()
        max_weight = (min_prob * self.size()) ** (beta)

        for i in range(batch_size):
            idx = indices[i]
            prob = self.sum_prior[idx + self.buffer_size] / self.get_sum()
            weight = (prob * self.size()) ** (-beta)
            weights[i] = weight / max_weight

        return indices, weights, self._get_samples(indices, env)
        


    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:

        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        
        next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)
        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


    def get_sum(self):
        return self.sum_prior[1]
    

    def get_min(self):
        return self.min_prior[1]


    def reset(self):
        super().reset()
        self.n_steps_buffer = [deque(maxlen=self.n_steps) for _ in range(self.n_envs)]
        self.sum_prior = np.full((2 * self.buffer_size), 0., dtype=np.float32)
        self.min_prior = np.full((2 * self.buffer_size), np.inf, dtype=np.float32)
        self.max_prior = 1.0

    
