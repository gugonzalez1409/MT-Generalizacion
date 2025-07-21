import gym
import gym.spaces
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels):
        
        # bloque residual con batch normalization
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        #self.batch1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        #self.batch2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):

        identity = x
        out = self.conv1(x)
        #out = self.batch1(out)
        out = F.relu(out)
        out = self.conv2(out)
        #out = self.batch2(out)
        out += identity

        return F.relu(out)

class ConvSequence(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(ConvSequence, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        #self.batch = nn.BatchNorm2d(out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_block1 = ResidualBlock(out_channels)
        self.res_block2 = ResidualBlock(out_channels)

    def forward(self, x):

        out = self.conv(x)
        #out = self.batch(out)
        out = F.relu(out)
        out = self.max_pool(out)
        out = self.res_block1(out)
        out = self.res_block2(out)

        return out

class ImpalaCNN(BaseFeaturesExtractor):

    def __init__(self, observation_spaces: gym.spaces.Box, features_dim = 256, depths=[16, 32, 32], scale = 1):

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

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(current_channels, features_dim)
        #self.dropout = nn.Dropout(p=0.1)

        self._init_weights()


    def forward(self, x):

        out = x.float() / 255.0
        out = self.conv_sequences(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        #out = self.dropout(out)
        out = self.fc(out)
        out = F.relu(out)
        
        return out
    
    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)