## archivo para tunning de hiperparametros

import gym
from gym.envs.registration import register
from .envs import make_single_env

def create_mario_env():

    return make_single_env(explore=False, random=False, custom=True, icm=False)


register(
    id='SMB-Tunning-v0',
    entry_point='models.utils.mario_env:create_mario_env',
)