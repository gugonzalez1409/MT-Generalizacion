import gym
import gym.spaces
import torch as th
from torch import nn
from .NoisyLinear import NoisyLinear
from sb3_contrib.qrdqn.policies import QRDQNPolicy
from typing import Any, Dict, List, Optional, Type
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN, create_mlp


class RainbowNet(BasePolicy):

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor: nn.Module,
        features_dim: int,
        n_quantiles: int = 200,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        dueling: bool = True,
        noisy: bool = True,
        noisy_kwargs: Optional[Dict[str, Any]] = None,

    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64, 64]

        
        self.dueling = dueling
        self.noisy = noisy
        self.noisy_kwargs = noisy_kwargs if noisy_kwargs is not None else {}
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        self.n_quantiles = n_quantiles
        self.normalize_images = normalize_images
        action_dim = self.action_space.n


        layers_class = nn.Linear if not self.noisy else NoisyLinear


        if self.dueling:
            # streams de valor y ventaja de dueling
            self.value_net = nn.Sequential(
                layers_class(self.features_dim, self.net_arch[0], **self.noisy_kwargs),
                self.activation_fn(),
                layers_class(self.net_arch[0], self.n_quantiles, **self.noisy_kwargs),
            )

            self.advantage_net = nn.Sequential(
                layers_class(self.features_dim, self.net_arch[0], **self.noisy_kwargs),
                self.activation_fn(),
                layers_class(self.net_arch[0], action_dim * self.n_quantiles, **self.noisy_kwargs),
            )

        else:
            # red sin dueling
            quantile_net = create_mlp(self.features_dim, action_dim * self.n_quantiles, self.net_arch, self.activation_fn)
            self.quantile_net = nn.Sequential(*quantile_net)


    def forward(self, obs: th.Tensor) -> th.Tensor:

        features = self.extract_features(obs)

        if self.dueling:

            value = self.value_net(features).view(-1, 1, self.n_quantiles)
            advantage = self.advantage_net(features).view(-1, self.action_space.n, self.n_quantiles)
            quantiles = value + advantage - advantage.mean(dim=1, keepdim=True)

        else:

            quantiles = self.quantile_net(features).view(-1, self.action_space.n, self.n_quantiles)
        
        return quantiles.permute(0, 2, 1) # batch, quantiles, actions
    
    
    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:

        q_values = self(observation).mean(dim=1)
        action = q_values.argmax(dim=1).reshape(-1)
        
        return action
    

    def reset_noise(self) -> None:

        if self.noisy:
            # reseteo de noise en ambos streams de dueling
            if self.dueling:
                for module in self.value_net.modules():
                    if isinstance(module, NoisyLinear):
                        module.reset_noise()
                for module in self.advantage_net.modules():
                    if isinstance(module, NoisyLinear):
                        module.reset_noise()
            else:
                # en caso de no usar dueling
                for module in self.quantile_net.modules():
                    if isinstance(module, NoisyLinear):
                        module.reset_noise()


class RainbowPolicy(QRDQNPolicy):

    def __init__(
      self,
      observation_space: gym.spaces.Space,
      action_space: gym.spaces.Space,
      lr_schedule: Schedule,
      n_quantiles: int = 200,
      net_arch: Optional[List[int]] = None,
      activation_fn: Type[nn.Module] = nn.ReLU,
      features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
      features_extractor_kwargs: Optional[Dict[str, Any]] = None,
      normalize_images: bool = True,
      optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
      optimizer_kwargs: Optional[Dict[str, Any]] = None,
      dueling: bool = True,
      noisy: bool = True,
      noisy_kwargs: Dict[str, Any] = None,

    ):
        self.dueling = dueling
        self.noisy = noisy
        self.noisy_kwargs = noisy_kwargs if noisy_kwargs is not None else {}
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            n_quantiles,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


    def make_quantile_net(self) -> RainbowNet:
        
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        net_args.update({
            'dueling': self.dueling,
            'noisy': self.noisy,
            'noisy_kwargs': self.noisy_kwargs
        })
        return RainbowNet(**net_args).to(self.device)
