import gym
import numpy as np


class customReward(gym.Wrapper):
    def __init__(self, env):
        super(customReward, self).__init__(env)

        # state: small, tall, fireball
        self.status = 'small'
        # posicion de step anterior
        self.prev_x_pos = 0
        # contador de tiempo estancado
        self.stuck_time = 0
    def reset(self, **kwargs):

        # reiniciar variables
        self.status = 'small'
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

            # si estuvo estancado por mas de 30 frames,y tiene un avance significativo 
            if self.stuck_time > 30 and (curr_x - self.prev_x_pos) > 3:
                reward += min(0.5 * self.stuck_time, 5.0)

            self.stuck_time = 0    
            # reiniciar contador de estancado

        # castigar perder el powerup
        if(self.status in ['tall', 'fireball'] and info['status'] == 'small'):
            reward -= 2.0

        self.status = info['status']

        # premiar completar nivel
        if done:
            if info['flag_get']:
                reward += 30.0

        self.prev_x_pos = curr_x

        return state, reward, done, info
    