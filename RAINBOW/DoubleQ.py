from sb3_contrib import QRDQN
from sb3_contrib.common.utils import quantile_huber_loss
import torch as th
import numpy as np

class DoubleQR(QRDQN):
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                next_q_values = self.quantile_net(replay_data.next_observations)
                next_q_values_mean = next_q_values.mean(dim=2)
                best_actions = next_q_values_mean.argmax(dim=1, keepdim=True)

                next_q_values_target = self.quantile_net_target(replay_data.next_observations)
                next_q_values_selected = next_q_values_target.gather(
                    dim=1,
                    index=best_actions.unsqueeze(-1).expand(-1, -1, next_q_values.shape[2])
                ).squeeze(dim=1)

                target_quantiles = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_selected

            current_quantiles = self.quantile_net(replay_data.observations)

            actions = replay_data.actions[..., None].long().expand(batch_size, self.n_quantiles, 1)
            current_quantiles = th.gather(current_quantiles, dim=2, index=actions).squeeze(dim=2)

            loss = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=True)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))