from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import NatureCNN
from ..generalization.ImpalaCNN import ImpalaCNN
from ..generalization.test_net import ValueNet
from typing import Dict, Any, Optional, List, Union, Type
from gym import spaces
import torch.nn as nn

class CustomPolicy(ActorCriticPolicy):
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs
    ):
        kwargs["share_features_extractor"] = False
        kwargs["features_extractor_class"] = NatureCNN
        kwargs["features_extractor_kwargs"] = {"features_dim": 512}
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs
        )
    
    def _build_mlp_extractor(self) -> None:

        self.pi_features_extractor = NatureCNN(
            self.observation_space, 
            features_dim=512
        )

        self.vf_features_extractor = ImpalaCNN(
            self.observation_space, 
            features_dim=512,
        )
        
        self.features_dim = 512
    
        super()._build_mlp_extractor()