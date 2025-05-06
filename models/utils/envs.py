import gym
from ..icm.ICM import ICM
from .reward import customReward
from ..icm.ICMneural import ICMneural
from .level_monitor import LevelMonitor
from nes_py.wrappers import JoypadSpace
from ..generalization.ExploreGo import ExploreGo
from ..generalization.DomainRand import DomainRandom
from stable_baselines3.common.atari_wrappers import AtariWrapper
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv, VecMonitor

tensorboard_log = r'./models/statistics/tensorboard_log/'
log_dir = r'./models/statistics/log_dir/'

# No incluye 7-4, 4-4, ni 8-4

ALL_LEVEL_LIST = [
        "1-1", "1-2", "1-3", "1-4",
        "2-1", "2-2", "2-3", "2-4",
        "3-1", "3-2", "3-3", "3-4",
        "4-1", "4-2", "4-3", "5-1", 
        "5-2", "5-3", "5-4", "6-1", 
        "6-2", "6-3", "6-4", "7-1", 
        "7-2", "7-3", "8-1", "8-2", 
        "8-3" ]


TRAINING_LEVEL_LIST = [
        "1-3", "2-1", "3-2", "3-4", 
        "4-2", "4-3", "5-3", "5-4", 
        "6-2", "6-4", "7-1", "7-2", 
        "7-3", "8-1", "8-3" ]

EVALUATION_LEVEL_LIST = [
        "1-1", "1-2", "1-4", "2-2",
        "2-3", "2-4", "3-1", "3-3", 
        "4-1", "5-1", "5-2", "6-1",
        "6-3", "8-2" ]


"""
Funciones de creacion de entorno SMB

"""

def eval_env(custom):
    """Entorno para EvalCallback"""

    env = gym.make('SuperMarioBrosRandomStages-v0', stages= EVALUATION_LEVEL_LIST)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    if custom: env = customReward(env)
    env = AtariWrapper(env=env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, clip_reward= False)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    env = VecMonitor(env)

    return env



def make_single_env(explore, random, custom):
    """Entorno simple para SMB"""

    env = gym.make('SuperMarioBrosRandomStages-v0', stages= TRAINING_LEVEL_LIST)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = AtariWrapper(env=env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, clip_reward= False)

    if(explore): env = ExploreGo(env, explore)
    if(random): env = DomainRandom(env, random)
    if(custom): env = customReward(env)

    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    env = VecMonitor(env, filename=log_dir)
    return env


"""Entorno vectorizado a numero de cores de CPU"""
def vectorizedEnv(explore, random, custom, icm = False):

    def make_env(explore, random, custom):

        env = gym.make('SuperMarioBrosRandomStages-v0', stages= TRAINING_LEVEL_LIST)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = AtariWrapper(env=env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, clip_reward= False)

        if(explore is not None): env = ExploreGo(env, explore)
        if(random): env = DomainRandom(env, random)
        if(custom): env = customReward(env)

        return env
    
    num_envs = 11
    env = VecMonitor(SubprocVecEnv([lambda: make_env(explore, random, custom) for _ in range(num_envs)]), filename=log_dir)
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    env = LevelMonitor(env)

    if(icm):
        observation_space = env.observation_space.shape
        action_space = env.action_space.n
        icm_model = ICMneural(observation_space, action_space)
        env = ICM(env, icm_model, update_interval=128)

    return env