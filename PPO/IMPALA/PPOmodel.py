import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv # correr varios envs en paralelo
from callback import SaveOnBestTrainingRewardCallback # callback para guardar modelo
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.atari_wrappers import AtariWrapper # wrappers estandar para atari/nes etc.
from nes_py.wrappers import JoypadSpace
import multiprocessing
from ImpalaCNN import Impala
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

"""
Agente base PPO para SMB
SB3 v1.6
Gym v0.21.0
Gym SMB v7.4.0
Pytorch 2.6.0
"""

def make_env():

    env = gym.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    # wrapper convencional para juegos maximo de NOOPS, salto de 4 frames, obs de 84x84x1 
    env = AtariWrapper(env=env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, clip_reward= False)

    return env

if __name__ == '__main__':

    # directorios para logs
    tensorboard_log = r'./tensorboard_log/'
    log_dir = r'./log_dir/'

    # numero de envs en paralelo
    num_envs = multiprocessing.cpu_count()-1

    policy_kwargs = dict(
    features_extractor_class=Impala,
    features_extractor_kwargs=dict(channels_num=4),
    )

    env = VecMonitor(SubprocVecEnv([make_env for _ in range(num_envs)]), filename=log_dir)
    
    # parametros PPO
    model = PPO(
        "CnnPolicy", 
        env, 
        policy_kwargs= policy_kwargs,
        learning_rate= 1e-6, 
        verbose=1,
        tensorboard_log = tensorboard_log)
    
    # guarda cada 10k pasos si la recompensa es mejor
    callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir)

    model.learn(total_timesteps=25000, callback=callback) # 5M  timesteps
    model.save("PPO_mario")