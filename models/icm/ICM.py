import gym
import torch
import torch.nn.functional as F

# calcula la accion con mayor intrinsic reward

class ICM:
    def __init__(self, icm_model, action_space):

        self.icm_model = icm_model
        self.action_space = action_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def select_action(self, obs):

        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0) / 255.0

        # extraer caracteristicas
        _obs = self.icm_model.feature_extractor(obs)

        max_curiosity = -float("inf")
        best_action = None

        for a in range(self.action_space.n):
            
            action_tensor = torch.tensor([a], device=self.device)
            action_onehot = F.one_hot(action_tensor, num_classes=self.action_space.n).float()

            forward_input = torch.cat([_obs, action_onehot], dim=1)
            phi_next_pred = self.icm.forward_model(forward_input)

            curiosity = F.mse_loss(phi_next_pred, _obs, reduction='sum').item()

            if curiosity > max_curiosity:
                max_curiosity = curiosity
                best_action = a

        return best_action