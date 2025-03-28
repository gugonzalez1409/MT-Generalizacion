import gym

# recompensa clippeada entre -15 y 15 (diferencia entre pos actual y anterior)
class customReward(gym.RewardWrapper):
    def __init__(self, env):
        super(customReward, self).__init__(env)

        self.score = 0
        # state: small, tall, fireball
        self.status = 'small'
        self.coins = 0

    def step(self, action):
        #self.env.render()
        state, reward, done, info = self.env.step(action)
        #reward += (info['score'] - self.score)
        #self.score = info['score']

        if(self.coins < info['coins']):
            self.coins = info['coins']
            reward += 1.0

        # premiar el conseguir powerups
        if(self.status != info['status']):
            self.status = info['status']
            reward += 2.0

        # castigar perder el powerup
        elif(self.status == 'tall' and info['status'] == 'small'):
            self.status = info['status']
            reward -= 2.0

        # premiar completar nivel
        if done:
            if info['flag_get']:
                reward += 30.0
                
        # castigo al morir o time over
            else:
                reward -= 30.0

        return state, reward / 10, done, info
