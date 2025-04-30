import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import gym
import torch as th
from sb3_contrib import QRDQN
from stable_baselines3 import DQN
from rainbow import Rainbow
from typing import Callable
from env_render import EnvRender, LevelFinish
from policies import RainbowPolicy
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.vec_env import VecMonitor
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.atari_wrappers import AtariWrapper


"""
Agente base DQN para SMB
SB3 v1.6
Gym v0.21.0
Gym SMB v7.4.0
Pytorch 2.6.0
"""

# de los docs de sb3
def linear_schedule(initial_value: float) -> Callable[[float], float]:

    def func(progress_remaining: float) -> float:

        return progress_remaining * initial_value

    return func


def make_env():
    # creacion del entorno
    env = gym.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = AtariWrapper(env=env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, clip_reward= False)
    return env

if __name__ == '__main__':

    device = "cuda" if th.cuda.is_available() else "cpu"

    tensorboard_log = r'./tensorboard_log/'
    log_dir = r'./log_dir/'

    num_envs = 11

    env = VecMonitor(SubprocVecEnv([make_env for _ in range(num_envs)]), filename=log_dir)
    env = VecFrameStack(env, n_stack=4, channels_order='last')

    model = Rainbow(
        RainbowPolicy,
        env,
        learning_rate=linear_schedule(2.5e-5),
        learning_starts=50000,
        policy_kwargs=dict(
            n_quantiles=200,
            net_arch=[256, 256],
            dueling=True,
            noisy=True,
            noisy_kwargs={
                'sigma': 0.5
            }
            ),
        buffer_size=100000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        exploration_fraction=0.4,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10000,
        verbose=1,
        device = device,
        tensorboard_log = tensorboard_log)
    
    model.learn(total_timesteps=50e6)
    model.save("RDQN_mario")