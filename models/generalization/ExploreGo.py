import gym
import random


class ExploreGo(gym.Wrapper):
    
    def __init__(self, env, exploration_steps, explorer):

        super(ExploreGo, self).__init__(env)
        self.exploration_steps = exploration_steps
        self.explorer = explorer

    def reset(self):

        obs = self.env.reset()
        for _ in range(random.randint(0, self.exploration_steps)):

            if self.explorer is not None:

                action = self.explorer.select_action(obs)

            else:

                action = self.env.action_space.sample()


            next_obs, reward, done, info = self.env.step(action)


            if self.explorer is not None:

                int_reward, loss = self.explorer.get_intrinsic_reward(obs, next_obs, action)
                self.explorer.update(loss)

            if done:

                obs = self.env.reset()

            else:

                obs = next_obs

        return obs

