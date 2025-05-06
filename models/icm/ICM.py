import gym
import torch

# calcula la accion con mayor intrinsic reward

class ICM:
    def __init__(self, icm_model, action_space):

        self.icm_model = icm_model
        self.action_space = action_space

