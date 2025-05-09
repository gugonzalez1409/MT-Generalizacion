import gym
import random


class ExploreGo(gym.Wrapper):
    
    def __init__(self, env, exploration_steps, explorer):

        super(ExploreGo, self).__init__(env)
        self.exploration_steps = exploration_steps
        self.explorer = explorer

    def reset(self):

        obs = self.env.reset()
        for i in range(random.randint(0, self.exploration_steps)):

            if self.explorer is not None:
                print("seleccionando accion en ICM")
                action = self.explorer.select_action(obs)

            else:
                #print("obs shape:", obs.shape) # 84,84,1
                action = self.env.action_space.sample()

            #print("accion explore-go: ",action)

            obs, reward, done, info = self.env.step(action)

            if done:

                obs = self.env.reset()

        return obs
    
