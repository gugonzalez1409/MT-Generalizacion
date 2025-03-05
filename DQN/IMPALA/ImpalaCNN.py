import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class Block(nn.Module):
    def __init__(self, channels):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        return self.relu(x + residual)

class Impala(BaseFeaturesExtractor):
    def __init__(self, observation_space, channels_num=1):
        super(Impala, self).__init__(observation_space, features_dim=256 * channels_num)

        c1, c2, c3 = [16, 32, 32]
        c1, c2, c3 = [c * channels_num for c in [c1, c2, c3]]

        self.conv1 = nn.Conv2d(observation_space.shape[0], c1, kernel_size=3, stride=1, padding=1)
        self.res1 = Block(c1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.res2 = Block(c2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.res3 = Block(c3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        with torch.no_grad():
            dummy_input = torch.zeros(1, *observation_space.shape)
            dummy_out = self._get_conv_output(dummy_input)
            self.flatten_dim = dummy_out.numel()

        self.fc = nn.Linear(self.flatten_dim, 256 * channels_num)


    def _get_conv_output(self, x):

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.res1(x)

        x = self.pool2(F.relu(self.conv2(x)))
        x = self.res2(x)

        x = self.pool3(F.relu(self.conv3(x)))
        x = self.res3(x)

        return torch.flatten(x, start_dim=1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.res1(x)

        x = self.pool2(F.relu(self.conv2(x)))
        x = self.res2(x)

        x = self.pool3(F.relu(self.conv3(x)))
        x = self.res3(x)
        
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc(x))
        return x
