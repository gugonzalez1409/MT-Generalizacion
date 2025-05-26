import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.atari_wrappers import AtariWrapper
from nes_py.wrappers import JoypadSpace


class E3BPolicy(nn.Module):

    def __init__(self, obs_shape, action_dim):

        super(E3BPolicy, self).__init__()
        c = obs_shape[2]
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # policy net para VizDoom segun paper "Exploration via Elliptical Episodic Bonuses"
        self.conv_layers = nn.Sequential(

            nn.AvgPool2d(2, 2),
            nn.Conv2d(c, 32, kernel_size=(3, 3), stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1),
            nn.ELU(),
            nn.Flatten()

        )

        self.lstm = nn.LSTM(

            input_size = 288,
            hidden_size = 256,
            num_layers = 2,

        )

        # obtenido de repo E3B: github.com/facebookresearch/e3b
        self.policy = nn.Sequential( nn.Linear(256, action_dim) )
        self.baseline = nn.Sequential( nn.Linear(256, 1) )

        # modelo inverso
        self.inverse_model = nn.Sequential(

            nn.Linear(2* 288, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim)

        )

    def forward(self):
        pass


class E3BEmbedding(nn.Module):

    def __init__(self, obs_shape):

        super(E3BEmbedding, self).__init__()
        # deberia guardar embeddings por nivel, para evitar recompensas muy altas
        # tras morir y pasar a nivel nuevo

        c = obs_shape[2]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # modelo de embedding igual que policy, pero sin lstm
        self.embedding = nn.Sequential(

            nn.AvgPool2d(2, 2),
            nn.Conv2d(c, 32, kernel_size=(3, 3), stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=1),
            nn.ELU(),
            nn.Flatten()

        )
    
    def forward(self):
        pass


class E3BInverse(nn.Module):

    def __init__(self, action_dim):

        super(E3BInverse, self).__init__()
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # modelo inverso, predice la accion
        self.inverse_model = nn.Sequential(

            nn.Linear(2 * 288, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim)

        )

    def forward(self, state, next_state):

        # obtener input para el modelo inverso
        states = torch.cat((state, next_state), dim=2) # revisar dimension
        pred_action = self.inverse_model(states)

        return pred_action


