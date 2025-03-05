import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import gym_super_mario_bros
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper

model = DQN.load("DQN_mario.zip")

def main():
    env = gym.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = AtariWrapper(env=env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, clip_reward=False)
    obs = env.reset()
    done = False

    while not done:
        action, state = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            env = env.reset()
            print("Reward: ", reward)


if __name__ == '__main__':
    main()
