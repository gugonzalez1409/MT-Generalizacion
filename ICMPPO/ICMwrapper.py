import gym
import torch

class ICMwrapper(gym.Wrapper):
    def __init__(self, env, icm, update_interval= 20):
        super(ICMwrapper, self).__init__(env)
        self.icm = icm # modelo ICM
        self.last_obs = None # guarda la observacion anterior
        self.update_interval = update_interval # intervalo para optimizar ICM
        self.count = 0

    def reset(self):
        self.last_obs = self.env.reset()
        return self.last_obs
    
    def step(self, action):
        #self.env.render()

        # actualmente ignora recompensa del entorno
        obs, reward, done, info = self.env.step(action)
        if self.last_obs is not None:
            # Convertir obs y next_obs a tensores normalizados (batch, C, H, W)
            obs_tensor = torch.tensor(self.last_obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            next_obs_tensor = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            action_tensor = torch.tensor([action], dtype=torch.int64)

            # Calcular la recompensa intrínseca y sumarla a la recompensa externa
            intrinsic_reward, loss = self.icm.get_intrinsic_reward(obs_tensor, next_obs_tensor, action_tensor)
            #print("intrinsic_reward:", intrinsic_reward)
            reward = intrinsic_reward
        
        self.last_obs = obs if not done else None

        self.count += 1
        if(self.count%20==0 or done == True):
            self.icm.update(loss)
            self.count = 0

        return obs, reward, done, info
