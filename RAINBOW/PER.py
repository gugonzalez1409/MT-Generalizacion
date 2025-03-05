from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np

class PER(ReplayBuffer):
    def __init__(self, *args, alpha=0.6, beta_start=0.4, beta_frames=100000, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha # controla la priorizacion
        self.beta_start = beta_start # controla la importancia de la correccion de sesgo
        self.beta_frames = beta_frames
        self.priorities = np.zeros((self.buffer_size,), dtype=np.float32)
        self.frame = 1 # contador para ajustar beta

    def add(self, *args, **kwargs):
        # añade experiencia al buffer
        #if len(args) > 0:
            #print("PER add - observation shape:", args[0].shape)
        max_prio = self.priorities.max() if self.size() > 0 else 1.0
        index = self.pos
        super().add(*args, **kwargs)
        self.priorities[index] = max_prio

    def sample(self, batch_size, env):

        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame +=1
        # 
        probs = self.priorities[:self.size()] ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(self.size(), batch_size, p=probs)
        #
        weights = (self.size() * probs[indices]) ** (-beta)
        weights /= weights.max()

        samples = super().sample(batch_size, env=env)

        return samples
    
    def update_priorities(self, indices, td_errors, epsilon=1e-5):
        self.priorities[indices] = np.abs(td_errors) + epsilon


class Nsteps(PER):
    def __init__(self, *args, n_step=3, gamma=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = []
    
    def add(self, obs, action, reward, next_obs, done, info):

        #print("obs:", next_obs.shape) #(11, 1, 84, 84)
        transition = (obs, action, reward, next_obs, done, info)
        self.buffer.append(transition)

        if len(self.buffer) < self.n_step:
            return
        
        R = sum([self.buffer[i][2] * (self.gamma ** i) for i in range(self.n_step)])
        state, action, _, next_state, done, info = self.buffer.pop(0)

        print("Adding to buffer - State shape:", state.shape)
        print("Adding to buffer - Next state shape:", next_state.shape)
        
        super().add(state, action, R, next_state, done, info)

    def sample(self, batch_size, env):
        samples = super().sample(batch_size, env = env)
        if samples is not None:
            print("Sampled observations shape:", samples.observations.shape)
            print("Sampled next observations shape:", samples.next_observations.shape)
        return samples
