from gym import RewardWrapper


class customReward(RewardWrapper):
    def __init__(self, env):
        super(customReward, self).__init__(env)

        self.score = 0
        # state: small, tall, fireball
        self.status = 'small'
        # posicion de step anterior
        self.prev_x_pos = 0
        # contador de tiempo estancado
        self.stuck_time = 0

    def step(self, action):

        state, reward, done, info = self.env.step(action)

        reward += (info['score'] - self.score) / 10
        self.score = info['score']

        # posicion actual
        curr_x = info['x_pos']


        # iniciar posicion en el primer step
        if self.prev_x_pos == 0:
            self.prev_x_pos = curr_x

        # en caso de retroceder, penalización mayor, no acumulativa
        if curr_x < self.prev_x_pos:
            reward -= 2.0
        
        # en caso de estar estancado, penalizacion acumulativa
        elif curr_x == self.prev_x_pos:
            self.stuck_time += 1
            reward -= 0.1 * self.stuck_time

        # en caso de avanzar, premiar, en caso de avanzar luego de
        # muchos steps estancado, premiar mas
        else:

            if self.stuck_time > 30:
                
                reward += 2.0
            
            else:

                reward += 1.0
                
            # reiniciar contador de estancado
            self.stuck_time = 0

        
        # castigar perder el powerup
        if(self.status in ['tall', 'fireball'] and info['status'] == 'small'):
            reward -= 2.0

        self.status = info['status']

        # premiar completar nivel
        if done:

            if info['flag_get']:
                reward += 30.0
                
        # castigo al morir o time over
            else:

                reward -= 30.0

        return state, reward / 10, done, info
