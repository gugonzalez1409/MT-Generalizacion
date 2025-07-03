import gym
from ..icm.ICM import ICMneural
from .reward import customReward
from ..icm.ICM import ICMneural
from .level_monitor import LevelMonitor
from nes_py.wrappers import JoypadSpace
from ..generalization.ExploreGo import ExploreGo
from ..generalization.DomainRand import DomainRandom
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv, VecMonitor, VecNormalize, VecTransposeImage

tensorboard_log = r'./models/statistics/tensorboard_log/'
log_dir = r'./models/statistics/log_dir/'


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
        "1-2", "1-4", "2-1", "2-3",       
        "3-2", "3-4", "4-1", "4-3",      
        "5-1", "5-4", "6-2", "6-4",       
        "7-1", "7-3", "8-2"
]

EVALUATION_LEVEL_LIST = [
        "1-1", "1-3", "2-2", "2-4",
        "3-1", "3-3", "4-2", "5-2", 
        "5-3", "6-1", "6-3", "7-2",              
        "8-1", "8-3"        
]


"""
Funciones de creacion de entorno SMB

"""

def make_single_env(explore, random, custom, icm):
    """Entorno simple para SMB"""

    env = gym.make('SuperMarioBrosRandomStages-v0', stages= TRAINING_LEVEL_LIST)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = AtariWrapper(env=env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, clip_reward= False)

    if(explore):
        if(icm):
            explorer = ICMneural(obs_shape=env.observation_space.shape, action_dim=env.action_space.n)
        else:
            explorer = None 
        env = ExploreGo(env, explore, explorer=explorer)
    if(random): env = DomainRandom(env, random, render=False) # usar render solo en entorno simple
    if(custom): env = customReward(env)

    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    env = VecMonitor(env, filename=log_dir)
    return env



def vectorizedEnv(explore, random, custom, icm = False, recurrent = False):
    """Entorno vectorizado a numero de cores de CPU"""
    def make_env(explore, random, custom):

        env = gym.make('SuperMarioBrosRandomStages-v1', stages= TRAINING_LEVEL_LIST)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = AtariWrapper(env=env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, clip_reward= False)

        if(explore is not None):
            if icm:
                explorer = ICMneural(obs_shape=env.observation_space.shape, action_dim=env.action_space.n) 
            else:
                explorer = None
            env = ExploreGo(env, explore, explorer=explorer)
        if(random): env = DomainRandom(env, random, render=False)
        if(custom): env = customReward(env)

        return env
    
    num_envs = 11
    env = VecMonitor(SubprocVecEnv([lambda: make_env(explore, random, custom) for _ in range(num_envs)]), filename=log_dir)

    if not recurrent:
        env = VecFrameStack(env, n_stack=4, channels_order='last')
    env = LevelMonitor(env)

    return env