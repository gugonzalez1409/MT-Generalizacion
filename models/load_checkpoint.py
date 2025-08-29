import gym
from stable_baselines3 import PPO
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, WarpFrame
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor
from utils.reward import customReward

log_dir = r'./models/'

TRAINING_LEVEL_LIST = ['1-2', '1-4', '2-1', '2-3', '3-2', '3-4', '4-1', '4-3', '5-1', '5-4', '6-2', '6-4', '7-1', '8-2']

def vectorizedEnv(explore, random, custom, icm = False, recurrent = False):

    def make_env(random, custom):

        env = gym.make('SuperMarioBrosRandomStages-v1', stages= TRAINING_LEVEL_LIST)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = MaxAndSkipEnv(env, skip=4) # frameskip de 4
        env = WarpFrame(env, width=84, height=84) # grayscale y resize
        if(custom): env = customReward(env)

        return env
    
    num_envs = 4 if explore else 11
    env = VecMonitor(SubprocVecEnv([lambda: make_env(random, custom) for _ in range(num_envs)]), filename=log_dir)

    if not recurrent:
        env = VecFrameStack(env, n_stack=4, channels_order='last')

    return env

if __name__ == "__main__":

    model = PPO.load("PPO_mario.zip", env=vectorizedEnv(explore=False, random=False, custom=True, icm=False, recurrent=False))
    model.learn(total_timesteps=1000000)