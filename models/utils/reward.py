import gym


class customReward(gym.Wrapper):
    def __init__(self, env):
        super(customReward, self).__init__(env)

        # posicion de step anterior
        self.prev_x_pos = 0
        # contador de tiempo estancado
        self.stuck_time = 0
    def reset(self, **kwargs):

        self.prev_x_pos = None
        self.stuck_time = 0

        return self.env.reset(**kwargs)

    def step(self, action):

        state, reward, done, info = self.env.step(action)
        curr_x = info['x_pos'] # posicion actual

        if self.prev_x_pos is None:
            self.prev_x_pos = curr_x

        # en caso de estar estancado, penalizacion acumulativa
        if curr_x == self.prev_x_pos:
            self.stuck_time += 1
            reward -= 0.2

        # premiar por superar estancamiento
        else:
            if self.stuck_time > 20:
                reward += 1.0

            self.stuck_time = 0

        # premiar completar nivel
        if done:
            if info['flag_get']:
                reward += 50.0

        self.prev_x_pos = curr_x

        return state, reward, done, info
    