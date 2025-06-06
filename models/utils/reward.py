import gym


class customReward(gym.Wrapper):
    def __init__(self, env):
        super(customReward, self).__init__(env)

        self.score = 0
        # state: small, tall, fireball
        self.status = 'small'
        # posicion de step anterior
        self.prev_x_pos = 0
        # contador de tiempo estancado
        self.stuck_time = 0

    def reset(self, **kwargs):

        # reiniciar variables
        self.score = 0
        self.status = 'small'
        self.prev_x_pos = 0
        self.stuck_time = 0

        return self.env.reset(**kwargs)

    def step(self, action):

        state, reward, done, info = self.env.step(action)
        reward += (info['score'] - self.score) / 10 # recompensar la diferencia positiva de puntaje
        self.score = info['score'] # asignar este puntaje como el actual

        
        curr_x = info['x_pos'] # posicion actual

        # iniciar posicion en el primer step
        if self.prev_x_pos == 0:
            self.prev_x_pos = curr_x

        # en caso de retroceder, penalización mayor, no acumulativa
        if curr_x < self.prev_x_pos:
            reward -= 2
        
        # en caso de estar estancado, penalizacion acumulativa
        elif curr_x == self.prev_x_pos:
            self.stuck_time += 1
            reward -= 0.1 * self.stuck_time

        # en caso de avanzar, premiar, en caso de avanzar luego de
        # muchos steps estancado, premiar mas
        else:
            if self.stuck_time > 30:
                reward += 2
                
            # reiniciar contador de estancado
            self.stuck_time = 0

        # castigar perder el powerup
        if(self.status in ['tall', 'fireball'] and info['status'] == 'small'):
            reward -= 2

        self.status = info['status']

        # premiar completar nivel
        if done:
            if info['flag_get']:
                reward += 300

        self.prev_x_pos = curr_x

        return state, reward / 10, done, info
    