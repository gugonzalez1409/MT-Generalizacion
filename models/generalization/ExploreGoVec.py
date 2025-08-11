import torch
import numpy as np
from typing import List, Tuple, Optional
from stable_baselines3.common.vec_env import VecEnvWrapper

class ExploreGoVec(VecEnvWrapper):

    def __init__(self, venv, K: int = 200, explorer: Optional[torch.nn.Module] = None):
        super().__init__(venv)
        self.K = K
        self.explorer = explorer
        self.random_steps_left = np.random.randint(0, self.K + 1, self.num_envs)
        self.current_obs = None

    def reset(self):
        
        obs = self.venv.reset()
        self.current_obs = obs
        self.random_steps_left = np.random.randint(0, self.K + 1, self.num_envs)

        return obs

    def step_async(self, actions: np.ndarray) -> None:

        final_actions = np.copy(actions)
        self.explore_step = [False] * self.num_envs

        for i in range(self.num_envs):

            if self.random_steps_left[i] > 0:
                self.explore_step[i] = True

                if self.explorer is not None:
                    with torch.no_grad():
                        obs_i = self.current_obs[i]
                        final_actions[i] = self.explorer.select_action(obs_i)
                else:
                    final_actions[i] = self.action_space.sample()
                self.random_steps_left[i] -= 1
        
        self.venv.step_async(final_actions)

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:

        obs, rewards, dones, infos = self.venv.step_wait()

        self.current_obs = obs

        for i in range(self.num_envs):
            infos[i]['is_explore_step'] = self.explore_step[i]
            if dones[i]:
                new_k = np.random.randint(0, self.K + 1)
                self.random_steps_left[i] = new_k
        
        return obs, rewards, dones, infos