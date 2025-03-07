import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import gym
import argparse
import multiprocessing
from icm.ICM import ICM
from icm.ICMneural import ICMneural
from stable_baselines3 import PPO, DQN
from nes_py.wrappers import JoypadSpace
from generalization.ImpalaCNN import Impala
from generalization.ExploreGo import ExploreGo
from generalization.DomainRand import DomainRandom
from stable_baselines3.common.vec_env import VecMonitor
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.atari_wrappers import AtariWrapper

"""
Agente base PPO para SMB
SB3 v1.6
Gym v0.21.0
Gym SMB v7.4.0
Pytorch 2.6.0
"""

LEVEL_LIST = [
        "1-1", "1-2", "1-3", "1-4",
        "2-1", "2-2", "2-3", "2-4",
        "3-1", "3-2", "3-3", "3-4",
        "4-1", "4-2", "4-3", "4-4",
        "5-1", "5-2", "5-3", "5-4",
        "6-1", "6-2", "6-3", "6-4",
        "7-1", "7-2", "7-3", "7-4",
        "8-1", "8-2", "8-3", "8-4"
        ]


def eval_env():

    env = gym.make('SuperMarioBrosRandomStages-v0', stages= LEVEL_LIST)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = AtariWrapper(env=env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, clip_reward= False)

    return env


def vectorizedEnv(explore, random, icm):

    def make_env(explore, random, icm):

        env = gym.make('SuperMarioBros-v0')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)

        if(explore): env = ExploreGo(env, explore)
        if(random): env = DomainRandom(env, random)
        if(icm):
            format = (4,84,84)
            icm = ICMneural(format, env.action_space.n)
            env = ICM(env, icm)

        env = AtariWrapper(env=env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, clip_reward= False)

        return env

    num_envs = multiprocessing.cpu_count()-1
    env = VecMonitor(SubprocVecEnv([lambda: make_env(explore, random, icm) for _ in range(num_envs)]), filename=log_dir)
    env = VecFrameStack(env, n_stack= 4)

    return env


def trainPPO(explore, random, impala, icm):

    policy_kwargs = {}

    if(impala):

        policy_kwargs = dict(
        features_extractor_class=Impala,
        features_extractor_kwargs=dict(channels_num=4),
        )

    model = PPO(
        "CnnPolicy",
        vectorizedEnv(explore, random, icm),
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log = tensorboard_log )
    
    callback = EvalCallback(eval_env=eval_env(), n_eval_episodes=100, eval_freq=100000,log_path=log_dir, best_model_save_path=log_dir)
    model.learn(total_timesteps=5e6, callback=callback)
    model.save("PPO_mario")


def trainDQN(explore, random, impala):

    policy_kwargs = {}

    if (impala):

        policy_kwargs = dict(
        features_extractor_class=Impala,
        features_extractor_kwargs=dict(channels_num=4)
        )
    
    model = DQN(
        "CnnPolicy", 
        vectorizedEnv(explore, random),
        policy_kwargs= policy_kwargs,
        buffer_size=100000,
        verbose=1,
        tensorboard_log = tensorboard_log )
    
    callback = EvalCallback(eval_env=eval_env(), n_eval_episodes=100, eval_freq=100000,log_path=log_dir, best_model_save_path=log_dir)
    model.learn(total_timesteps=5e6, callback=callback)
    model.save("DQN_mario")


def test(model_name):

    model = PPO.load(model_name)
    env = eval_env()
    obs = env.reset()

    for _ in range(10):
        obs = env.reset()
        while True:
            env.render()
            action, _ = model.predict(obs)
            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset()
                break


if __name__ == '__main__':
    
    tensorboard_log = r'./tensorboard_log/'
    log_dir = r'./log_dir/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help= 'Modo de ejecucion')
    parser.add_argument('--model', type=str, help= 'Nombre del modelo a cargar')
    parser.add_argument('--algo', type=str, choices=['PPO', 'DQN'], help= 'Algoritmo a entrenar')
    parser.add_argument('--explore', type=int, default=100, help= 'Pasos de exploracion en ExploreGo')
    parser.add_argument('--impala', action='store_true', default=False, help= 'Uso de arquitectura de red IMPALA')
    parser.add_argument('--random', type=int, default=50, help= 'Cantidad de frames en las que se usa randomizacion de entorno')
    parser.add_argument('--icm', action='store_true', default=False, help='En caso de entrenar con PPO, usa modulo de curiosidad intrinseca')

    args = parser.parse_args()

    if args.mode == 'train':

        if args.algo == None:
            parser.error('Se requiere especificar un algoritmo para entrenar')
        
        if args.algo == 'PPO':
            trainPPO(args.explore, args.random, args.impala, args.icm)

        if args.algo == 'DQN':
            if(args.icm == True):
                parser.error("Modulo de curiosidad exclusivo de PPO")
            trainDQN(args.explore, args.random, args.impala)
    
    if args.mode == 'test':

        test(args.model)
