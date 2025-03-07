import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ICMneural(nn.Module):
    def __init__(self, obs_shape, action_dim, feature_dim=256, lr=1e-3):
        super(ICMneural, self).__init__()
        # channel, height, width (4,84,84)
        c, _, _ = obs_shape

        # nn feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, feature_dim),
            nn.ReLU()
        )

        # modelo inverso, predice la accion
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

        # modelo directo, predice el siguiente estado
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

        # optimizador Adam y funcion de perdida MSE
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.loss_inv = nn.CrossEntropyLoss()

    def forward(self, state, next_state, action):

        # obtiene las caracteristicas de las observaciones
        state_ = self.feature_extractor(state)
        next_state_ = self.feature_extractor(next_state)

        pred_action = self.inverse_model(torch.cat((state_, next_state_), dim=1))

        action_onehot = torch.nn.functional.one_hot(action, num_classes=pred_action.shape[1]).float()
        forward_input = torch.cat((state_, action_onehot), dim=1)
        pred_next_state = self.forward_model(forward_input)

        return state_, next_state_, pred_action, pred_next_state

    def get_intrinsic_reward(self, state, next_state, action):
        # obtiene las predicciones y calcula la recompensa intrinseca como la diferencia entre las caracteristicas predichas y reales
        state_, next_state_, pred_action, pred_next_state = self.forward(state, next_state, action)
        intrinsic_reward = self.loss_fn(pred_next_state, next_state_).detach().cpu().numpy()

        inv_loss = self.loss_inv(pred_action, action)
        forw_loss = self.loss_fn(pred_next_state, next_state_)
        total_loss = inv_loss + forw_loss


        return intrinsic_reward, total_loss
    
    def update(self, loss):
        # actualiza el modelo periodicamente
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()