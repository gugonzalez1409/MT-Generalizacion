import gym

class EnvRender(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        self.env.render()
        return super().step(action)