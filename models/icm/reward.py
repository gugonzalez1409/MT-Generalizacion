import gym

# recompensa clippeada entre -15 y 15 (diferencia entre pos actual y anterior)
class customReward(gym.RewardWrapper):
    def __init__(self, env):
        super(customReward, self).__init__(env)

        self.score = 0
        # state: small, tall, fireball
        self.status = 'small'
        self.coins = 0

# recompensa por obtener puntos
    def step(self, action):
        #self.env.render()
        state, reward, done, info = self.env.step(action)
        #eward += (info['score'] - self.score)
        #self.score = info['score']

        if(self.coins < info['coins']):
            self.coins = info['coins']
            reward += 0.5

        # premiar el conseguir champiñon o flor de fuego
        if(self.status != info['status']):
            self.status = info['status']
            reward += 1.5

        # castigar el ser golpeado
        elif(self.status == 'tall' and info['status'] == 'small'):
            self.status = info['status']
            reward -= 1.5

        # premiar completar nivel
        if done:
            if info['flag_get']:
                reward += 30.0
                
        # castigo al morir o time over
            else:
                reward -= 30.0

        return state, reward / 15, done, info
