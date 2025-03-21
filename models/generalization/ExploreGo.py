import gym
import random


class ExploreGo(gym.Wrapper):
    # exploration steps: cantidad de pasos de exploracion por episodio
    def __init__(self, env, exploration_steps):
        super(ExploreGo, self).__init__(env)
        self.exploration_steps = exploration_steps

    # hace reset del entorno y toma exploration_steps acciones random
    def reset(self):
        obs = self.env.reset()
        for i in range(random.randint(0, self.exploration_steps)):
            action = self.env.action_space.sample()
            print("accion random: ", i)
            print(action)
            obs, _, done, _ = self.env.step(action)
            if done:
                obs = self.env.reset()
        return obs
    
