import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups, kernel_size=4, stride=2, padding=1):
        
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.group_norm(x)
        x = self.relu(x)
        return x
    


class ValueNet(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(ValueNet, self).__init__(observation_space, features_dim)

        n_channels = observation_space.shape[0]

        self.conv1 = ConvBlock(in_channels=n_channels, out_channels=64, num_groups=8)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128, num_groups=16)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256, num_groups=16)
        self.conv4 = ConvBlock(in_channels=256, out_channels=256, num_groups=16)

        with torch.no_grad():
            dummy_input = torch.zeros(1, *observation_space.shape)
            n_flatten = self._get_conv_output(dummy_input)

        self.flatten = nn.Flatten()

        self.linear = nn.Linear(n_flatten, features_dim)
        self.relu = nn.ReLU()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:

        x = self.conv1(observations)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.relu(x)

        return x

    def _get_conv_output(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x.numel()