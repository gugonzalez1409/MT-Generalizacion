import gym
import torch as th
import numpy as np
from .per import PER
from sb3_contrib import QRDQN
from .policies import RainbowPolicy
from sb3_contrib.common.utils import quantile_huber_loss
from stable_baselines3.common.buffers import ReplayBuffer
from typing import Any, Dict, Optional, Type, Union
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import polyak_update

"""

Implementación de Rainbow DQN, usando de base QRDQN de sb3-contrib.

"""

class Rainbow(QRDQN):

    def __init__(
        self,
        policy: Union[str, Type[RainbowPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 2.5e-5,
        buffer_size: int = 1000000,
        learning_starts: int = 10000,
        batch_size: int = 32,
        tau: float = 1.0,
        optimize_memory_usage: bool = False,
        create_eval_env: bool = False,
        gamma: float = 0.99,
        max_grad_norm: Optional[float] = None,
        train_freq: int = 4,
        gradient_steps: int = 2,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = PER,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.005,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.01,
        tensorboard_log: Optional[str] = None,
        # parametros para noisy linear
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = 'auto'
            
    ) -> None:
        
        if replay_buffer_kwargs is not None:
            replay_buffer_kwargs['gamma'] = gamma

        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            optimize_memory_usage=optimize_memory_usage,
        )



    def train(self, gradient_steps: int, batch_size: int = 100) -> None:

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []

        for _ in range(gradient_steps):
            
            # reset noise de noisy nets
            if hasattr(self.quantile_net, "reset_noise"):
                # reseteo de noise de red online
                self.quantile_net.reset_noise()

            indices, weights, replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # double q obtiene los q values de la red online y evalua con la red target
            with th.no_grad():
                
                next_q_values = self.quantile_net(replay_data.next_observations) # batch, quantiles, actions
                next_q_values_mean = next_q_values.mean(dim=1)  # batch, actions
                best_actions = next_q_values_mean.argmax(dim=1, keepdim=True) # batch, 1

                # reset de red target
                if hasattr(self.quantile_net_target, "reset_noise"):
                    self.quantile_net_target.reset_noise()

                next_q_values_target = self.quantile_net_target(replay_data.next_observations) # batch, quantiles, actions
                next_q_values_selected = next_q_values_target.gather(dim=2, index=best_actions.unsqueeze(1).expand(-1, next_q_values_target.shape[1], -1)).squeeze(2)

                target_quantiles = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_selected

            current_quantiles = self.quantile_net(replay_data.observations)

            actions = replay_data.actions[..., None].long().expand(batch_size, self.n_quantiles, 1)
            current_quantiles = th.gather(current_quantiles, dim=2, index=actions).squeeze(dim=2)

            # td errors para per
            td_errors = th.abs(current_quantiles.mean(dim=1) - target_quantiles.mean(dim=1)).detach().cpu().numpy()
            # update de prioridades per
            self.replay_buffer.update_priorities(indices, td_errors + 1e-6)

            # calcular loss
            error = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=True)
            loss = (th.as_tensor(weights, device=self.device) * error).mean()

            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


    def _on_step(self):
        # actualizar log de beta per
        self.replay_buffer.update_beta(self._current_progress_remaining)

        if self.num_timesteps % self.target_update_interval == 0:
            polyak_update(self.quantile_net.parameters(), self.quantile_net_target.parameters(), self.tau)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)
        self.logger.record("rollout/per_beta", self.replay_buffer.beta)

    def predict(self, observation, state = None, episode_start = None, deterministic = False):
        if hasattr(self.policy, "reset_noise"):
            #print("Reseteando noise en predict")
            self.policy.reset_noise()
        return super().predict(observation, state, episode_start, deterministic)

