import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import gym
import gym_super_mario_bros
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv # correr varios envs en paralelo
from callback import SaveOnBestTrainingRewardCallback # callback para guardar modelo
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.atari_wrappers import AtariWrapper # wrappers estandar para atari/nes etc.
from nes_py.wrappers import JoypadSpace
import multiprocessing
from ImpalaCNN import Impala
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

"""
Agente base DQN para SMB
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
    # directorios para guardar logs
    tensorboard_log = r'./tensorboard_log/'
    log_dir = r'./log_dir/'

    # crea la cantidad de entornos segun en numero de cpus
    num_envs = multiprocessing.cpu_count()-1

    # crea el vector de entornos
    env = VecMonitor(SubprocVecEnv([make_env for _ in range(num_envs)]), filename=log_dir)

    policy_kwargs = dict(
    features_extractor_class=Impala,
    features_extractor_kwargs=dict(channels_num=4),
    )

    # crea el modelo
    model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, buffer_size=100000, verbose=1, tensorboard_log = tensorboard_log)
    callback = SaveOnBestTrainingRewardCallback(check_freq=100000, log_dir=log_dir)
    model.learn(total_timesteps=5e6, callback=callback) # 5M  timesteps
    model.save("DQN_mario")