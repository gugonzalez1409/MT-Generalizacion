import math
import torch as th
from torch import nn

class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma: float = 0.5):
        
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma

        # mu y sigma param de la distribucion normal
        self.weight_mu = nn.Parameter(th.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(th.empty(out_features))

        self.weight_sigma = nn.Parameter(th.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(th.empty(out_features))

        self.register_buffer("weight_epsilon", th.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", th.empty(out_features))

        self.reset_parameters()
        self.reset_noise()


    def reset_parameters(self):

        mu_range = 1 / math.sqrt(self.in_features)

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma / math.sqrt(self.out_features))


    def reset_noise(self):
        
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(th.outer(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, features: int):

        x = th.randn(features)
        return x.sign().mul(x.abs().sqrt())
    
    def forward(self, input: th.Tensor) -> th.Tensor:

        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon

        return th.nn.functional.linear(input, weight, bias)