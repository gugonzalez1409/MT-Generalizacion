import gym
import gym.spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ResidualBlock(nn.Module):
    """
    Bloque residual utilizado en IMPALA CNN.
    """
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        identity = x
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += identity
        return out

class ConvSequence(nn.Module):
    """
    Secuencia Convolucional de IMPALA CNN
    Una capa convolucional, Max pooling y dos bloques residuales.
    """
    def __init__(self, in_channels, out_channels):

        super(ConvSequence, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_block1 = ResidualBlock(out_channels)
        self.res_block2 = ResidualBlock(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.max_pool(out)
        out = self.res_block1(out)
        out = self.res_block2(out)
        return out

class ImpalaCNN(BaseFeaturesExtractor):
    """
    Implementación de la IMPALA CNN
    """
    def __init__(self, observation_spaces: gym.spaces.Box, features_dim = 512, depths=[16, 32, 32], scale = 1):

        super().__init__(observation_spaces, features_dim=features_dim)
        
        in_channels = observation_spaces.shape[0]
        scaled_depths = [int(d * scale) for d in depths]
        self.depths = scaled_depths

        conv_seqs = []
        current_channels = in_channels
        for depth in scaled_depths:
            conv_seqs.append(ConvSequence(current_channels, depth))
            current_channels = depth
        
        self.conv_sequences = nn.Sequential(*conv_seqs)
        self.fc = None
        self._fc_initialized = False

        self._init_weights()


    def forward(self, x):

        out = x.float() / 255.0
        out = self.conv_sequences(out)
        out = torch.flatten(out, start_dim=1)
        out = F.relu(out)

        if not self._fc_initialized or self.fc.in_features != out.shape[1]:

            self.fc = nn.Linear(out.shape[1], self._features_dim).to(out.device)
            nn.init.xavier_uniform_(self.fc.weight)
            if self.fc.bias is not None:
                nn.init.zeros_(self.fc.bias)

            self.fc_initialized = True

        out = self.fc(out)
        out = F.relu(out)
        
        return out
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)