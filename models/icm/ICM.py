import gym
import torch

class ICM(gym.Wrapper):
    def __init__(self, env, icm, update_interval= 512):
        super(ICM, self).__init__(env)
        self.icm = icm 
        self.last_obs = None
        self.update_interval = update_interval
        self.losses = []
        self.count = 0

    def reset(self):
        self.last_obs = self.env.reset()
        return self.last_obs
    
    def step(self, action):
        #self.env.render()
        obs, reward, done, info = self.env.step(action)
        if self.last_obs is not None:

            obs_T = torch.tensor(self.last_obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            next_obs_T = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            action_T = torch.tensor(action, dtype=torch.long)

            intrinsic_reward, loss = self.icm.get_intrinsic_reward(obs_T, next_obs_T, action_T)
            self.losses.append(loss)
            reward += intrinsic_reward
        
        self.last_obs = obs if not done else None

        self.count += 1
        if(self.count%self.update_interval==0):

            total_loss = sum(self.losses) / len(self.losses)
            self.icm.update(total_loss)
            self.losses.clear()


        return obs, reward, done, info