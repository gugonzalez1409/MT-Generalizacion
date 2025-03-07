import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import gym
from DoubleQ import DoubleQR # Double
from DuelingNet import DuelingPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv # correr varios envs en paralelo
from stable_baselines3.common.vec_env import VecFrameStack
from callback import SaveOnBestTrainingRewardCallback # callback para guardar modelo
from stable_baselines3.common.vec_env import VecMonitor # monitoreo de entornos
from stable_baselines3.common.atari_wrappers import AtariWrapper
from nes_py.wrappers import JoypadSpace
import multiprocessing
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
    # wrapper convencional para juegos maximo de NOOPS, salto de 4 frames, obs de 84x84x1 SubprocVecEnv= (11,84,84,1)
    env = AtariWrapper(env=env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, clip_reward= False)
    return env

if __name__ == '__main__':

    tensorboard_log = r'./tensorboard_log/'
    log_dir = r'./log_dir/'

    num_envs = multiprocessing.cpu_count()-1

    env = VecMonitor(SubprocVecEnv([make_env for _ in range(num_envs)]), filename=log_dir)
    env = VecFrameStack(env, n_stack=4)
    obs = env.reset()

    model = DoubleQR(
        DuelingPolicy,
        env,
        learning_rate=1e-6,
        policy_kwargs=dict(n_quantiles=200),
        buffer_size=100000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        verbose=1,
        tensorboard_log = tensorboard_log)
    
    callback = SaveOnBestTrainingRewardCallback(check_freq=100000, log_dir=log_dir)
    model.learn(total_timesteps=5e6, callback=callback) # 5M  timesteps
    model.save("RDQN_mario")