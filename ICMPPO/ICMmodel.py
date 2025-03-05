import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv # correr varios envs en paralelo
from callback import SaveOnBestTrainingRewardCallback # callback para guardar el mejor modelo
from stable_baselines3.common.vec_env import VecMonitor # monitorea los entornos
from stable_baselines3.common.atari_wrappers import AtariWrapper # wrappers estandar para atari/nes etc.
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from ICMneural import ICM
from ICMwrapper import ICMwrapper
import multiprocessing



"""
Agente PPO para SMB
SB3 v1.6
Gym v0.21.0
Gym SMB v7.4.0
Pytorch 2.6.0
"""

format = (1,84,84)

def make_env():

    env = gym.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    # wrapper convencional para juegos maximo de NOOPS, salto de 4 frames, obs de 84x84x1
    env = AtariWrapper(env=env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, clip_reward= True)
    icm = ICM(format, env.action_space.n)
    env = ICMwrapper(env, icm)

    return env

if __name__ == '__main__':
    
    tensorboard_log = r'./tensorboard_log/'
    log_dir = r'./log_dir/'

    num_envs = multiprocessing.cpu_count()-1
    env = VecMonitor(SubprocVecEnv([make_env for _ in range(num_envs)]), filename=log_dir)

    model = PPO("CnnPolicy", env, learning_rate= 1e-6, verbose=1,tensorboard_log = tensorboard_log)
    callback = SaveOnBestTrainingRewardCallback(check_freq=100000, log_dir=log_dir)
    
    model.learn(total_timesteps=5e6, callback=callback) # 5M  timesteps
    model.save("ICM_mario")