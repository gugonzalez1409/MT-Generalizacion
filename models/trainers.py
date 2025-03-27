import gym
import numpy as np
import multiprocessing
from icm.ICM import ICM
from typing import Callable
from icm.reward import customReward
from icm.ICMneural import ICMneural
from stable_baselines3 import PPO, DQN
from nes_py.wrappers import JoypadSpace
from generalization.ExploreGo import ExploreGo
from generalization.DomainRand import DomainRandom
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.atari_wrappers import AtariWrapper



tensorboard_log = r'./statistics/tensorboard_log/'
log_dir = r'./statistics/log_dir/'

ALL_LEVEL_LIST = [
        "1-1", "1-2", "1-3", "1-4",
        "2-1", "2-2", "2-3", "2-4",
        "3-1", "3-2", "3-3", "3-4",
        "4-1", "4-2", "4-3", "4-4",
        "5-1", "5-2", "5-3", "5-4",
        "6-1", "6-2", "6-3", "6-4",
        "7-1", "7-2", "7-3", "7-4",
        "8-1", "8-2", "8-3", "8-4"
        ]

"""
Funciones de creacion de entorno SMB

"""


"""Entorno para EvalCallback"""
def eval_env():

    env = gym.make('SuperMarioBrosRandomStages-v0', stages= ALL_LEVEL_LIST)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = AtariWrapper(env=env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, clip_reward= False)
    env = Monitor(env, filename=log_dir)

    return env


"""Entorno simple para SMB"""
def make_single_env(explore, random, custom):

    env = gym.make('SuperMarioBrosRandomStages-v0', stages= ALL_LEVEL_LIST)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = AtariWrapper(env=env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, clip_reward= False)

    if(explore): env = ExploreGo(env, explore)
    if(random): env = DomainRandom(env, random)
    if(custom): env = customReward(env)

    env = Monitor(env, filename=log_dir)
    # entorno simple no compatible con wrappers ni de gym ni sb3 para framestack
    return env


"""Entorno vectorizado a numero de cores de CPU"""
def vectorizedEnv(explore, random, custom, icm = False):

    def make_env(explore, random, custom):

        env = gym.make('SuperMarioBrosRandomStages-v0', stages= ALL_LEVEL_LIST)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = AtariWrapper(env=env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, clip_reward= False)

        if(explore is not None):
            print("voy a explorar")
            env = ExploreGo(env, explore)
        if(random):
            print("randomizando entorno")
            env = DomainRandom(env, random)
        if(custom): 
            print("usando recompensa custom")
            env = customReward(env)

        return env
    
    num_envs = multiprocessing.cpu_count() - 1
    env = VecMonitor(SubprocVecEnv([lambda: make_env(explore, random, custom) for _ in range(num_envs)]), filename=log_dir)

    if(icm):
        observation_space = env.observation_space.shape
        action_space = env.action_space.n
        icm_model = ICMneural(observation_space, action_space)
        env = ICM(env, icm_model, update_interval=128)

    return env


"""Funciones de Scheduler para LR"""

def linear_schedule(initial_value: float ) -> Callable[[float], float]:
    
    def func(progress_remaining: float) -> float:

        return progress_remaining * initial_value
    

    return func


def schedule(initial_value: float, final_value: float) -> Callable[[float], float]:

    def func(progress_remaining: float) -> float:
        
        cos_decay = 0.5 * (1 + np.cos(np.pi * (1 - progress_remaining)))
        return final_value + (initial_value - final_value) * cos_decay
    
    return func


"""
Funciones de entrenamiento de agentes

"""


def trainPPO(explore, random, custom, vectorized):

    policy_kwargs = {}

    model = PPO(
        'CnnPolicy',
        learning_rate=schedule(2.5e-4, 1e-5),
        env = vectorizedEnv(explore, random, custom) if vectorized else make_single_env(explore, random, custom),
        policy_kwargs=policy_kwargs,
        ent_coef=0.03,
        verbose=1,
        tensorboard_log = tensorboard_log
        )
    
    callback = EvalCallback(eval_env=eval_env(), n_eval_episodes=100, eval_freq=100000,log_path=log_dir, best_model_save_path=log_dir)
    model.learn(total_timesteps=100e6, callback=callback)
    model.save("PPO_mario")




def trainDQN(explore, random, custom, vectorized):

    policy_kwargs = {}
    
    model = DQN(
        "CnnPolicy",
        learning_rate = schedule(1e-4,1e-3),
        env = vectorizedEnv(explore, random, custom) if vectorized else make_single_env(explore, random, custom),
        policy_kwargs= policy_kwargs,
        buffer_size=100000,
        learning_starts=50000,
        exploration_final_eps= 0.05,
        exploration_fraction=0.4,
        verbose=1,
        tensorboard_log = tensorboard_log
        )
    
    
    callback = EvalCallback(eval_env=eval_env(), n_eval_episodes=100, eval_freq=100000,log_path=log_dir, best_model_save_path=log_dir)
    model.learn(total_timesteps=100e6, callback=callback)
    model.save("DQN_mario")