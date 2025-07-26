import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, WarpFrame

def make_mario_env(env_id, stages=None):
    if stages:
        env = gym.make(env_id, stages=stages)
    else:
        env = gym.make(env_id)
    
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = WarpFrame(env)
    env = MaxAndSkipEnv(env, skip=4)
    
    return env