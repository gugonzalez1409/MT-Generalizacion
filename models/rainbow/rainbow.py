import gym
import torch as th
import numpy as np
from .per import PER
from sb3_contrib import QRDQN
from copy import deepcopy
from .policies import RainbowPolicy
from sb3_contrib.common.utils import quantile_huber_loss
from stable_baselines3.common.buffers import ReplayBuffer
from typing import Any, Dict, Optional, Type, Union
from stable_baselines3.common.type_aliases import GymEnv, Schedule, RolloutReturn
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.utils import should_collect_more_steps


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
        gradient_steps: int = 1,
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
            self.policy.reset_noise()
        return super().predict(observation, state, episode_start, deterministic)
    

    # modificado para usar ExploreGo
    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos):

        valid_mask = np.array([not info.get('is_explre_step', False) for info in infos])
        valid_indices = np.where(valid_mask)[0]

        total_steps = len(infos)
        valid_steps = len(valid_indices)
        discarded_steps = total_steps - valid_steps
        if discarded_steps > 0:
            print(f"Paso de entorno: {total_steps} transiciones totales. {discarded_steps} DESCARTADAS (exploración), {valid_steps} AÑADIDAS.")

        if len(valid_indices) == 0:
            return

        last_obs_filtered = self._last_obs[valid_indices]
        buffer_action_filtered = buffer_action[valid_indices]
        new_obs_filtered = new_obs[valid_indices]
        reward_filtered = reward[valid_indices]
        dones_filtered = dones[valid_indices]
        infos_filtered = [infos[i] for i in valid_indices]

        if self._vec_normalize_env is not None:

            new_obs_ = self._vec_normalize_env.get_original_obs()[valid_indices]
            reward_ = self._vec_normalize_env.get_original_reward()[valid_indices]
            last_original_obs_ = self._last_original_obs[valid_indices]
        else:
            last_original_obs_, new_obs_, reward_ = last_obs_filtered, new_obs_filtered, reward_filtered

        next_obs = deepcopy(new_obs_)

        for i, done in enumerate(dones_filtered):

            info_idx = valid_indices[i]
            if done and infos[info_idx].get("terminal_observation") is not None:
                terminal_obs = infos[info_idx]["terminal_observation"]
                if self._vec_normalize_env is not None:
                    terminal_obs = self._vec_normalize_env.unnormalize_obs(terminal_obs)
                
                if isinstance(next_obs, dict):
                    for key in next_obs.keys():
                        next_obs[key][i] = terminal_obs[key]
                else:
                    next_obs[i] = terminal_obs

        replay_buffer.add(
            last_original_obs_,
            next_obs,
            buffer_action_filtered,
            reward_,
            dones_filtered,
            infos_filtered,
        )


    # pequeña modificacion para contar pasos en entorno
    def collect_rollouts(self, env, callback, train_freq, replay_buffer, action_noise = None, learning_starts = 0, log_interval = None):
        
        self.policy.set_training_mode(False) # batch norm / dropout importante

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            valid_steps_count = sum(1 for info in infos if not info.get('is_explore_step', False))
            self.num_timesteps += valid_steps_count
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()

        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)