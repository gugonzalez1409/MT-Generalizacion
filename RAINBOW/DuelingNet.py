import torch as th
import torch.nn as nn
from sb3_contrib.qrdqn.policies import QuantileNetwork, QRDQNPolicy

class DuelingNetwork(QuantileNetwork):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        action_dim = self.action_space.n
        n_quantiles = self.n_quantiles

        self.value_stream = nn.Sequential(
            nn.Linear(self.features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_quantiles)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(self.features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim * n_quantiles)
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)
        x = self.quantile_net[:-1](features)

        V = self.value_stream(x).unsqueeze(1)
        A = self.advantage_stream(x).view(-1, self.action_space.n, self.n_quantiles)

        A_mean = A.mean(dim=1, keepdim=True)
        Q = V + (A - A_mean)
        return Q

class DuelingPolicy(QRDQNPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def make_quantile_net(self) -> QuantileNetwork:
        return DuelingNetwork(
            self.observation_space,
            self.action_space,
            self.features_extractor,
            self.features_dim,
            self.n_quantiles,
            self.net_arch,
            self.activation_fn,
            self.normalize_images,
        ).to(self.device)