import numpy as np
from typing import Callable
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO, DQN
from models.utils.envs import make_single_env, vectorizedEnv, eval_env
from stable_baselines3.common.callbacks import EvalCallback

tensorboard_log = r'./models/statistics/tensorboard_log/'
log_dir = r'./models/statistics/log_dir/'

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
    
    callback = EvalCallback(eval_env=eval_env(), n_eval_episodes=10, eval_freq=100000,log_path=log_dir, best_model_save_path=log_dir)
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
    
    
    callback = EvalCallback(eval_env=eval_env(), n_eval_episodes=10, eval_freq=100000,log_path=log_dir, best_model_save_path=log_dir)
    model.learn(total_timesteps=100e6, callback=callback)
    model.save("DQN_mario")



def trainRecurrentPPO(explore, random, custom, vectorized):

    policy_kwargs = {}

    model = RecurrentPPO(
        'CnnPolicy',
        learning_rate = schedule(2.5e-4, 1e-5),
        env = vectorizedEnv(explore, random, custom) if vectorized else make_single_env(explore, random, custom),
        policy_kwargs= policy_kwargs,
        ent_coef=0.03,
        verbose=1,
        tensorboard_log = tensorboard_log
    )

    callback = EvalCallback(eval_env = eval_env(), n_eval_episodes=10, eval_freq=100000, log_path=log_dir, best_model_save_path=log_dir)
    model.learn(total_timesteps=100e6, callback=callback)
    model.save("RecurrentPPO_mario")