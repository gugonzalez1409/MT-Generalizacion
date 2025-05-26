import torch
import torch.nn as nn
import torch.nn.functional as F
from E3Bneural import E3BPolicy, E3BEmbedding, E3BInverse

# clase para iniciar el modelo y optimizers

class E3B():

    def __init__(self, obs_shape, action_dim, lr=1e-3):

        self.policy = E3BPolicy(obs_shape=obs_shape, action_dim=action_dim)
        self.embedding = E3BEmbedding(obs_shape=obs_shape)
        self.inverse = E3BInverse(action_dim=action_dim)

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.embedding_optimizer = torch.optim.Adam(self.embedding.parameters(), lr=lr)
        self.inverse_optimizer = torch.optim.Adam(self.inverse.parameters(), lr=lr)


