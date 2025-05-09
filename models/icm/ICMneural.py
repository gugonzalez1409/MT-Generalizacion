import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


## hacer que el modelo retorne acciones discretas para Mario Bros


class ICMneural(nn.Module):
    def __init__(self, obs_shape, action_dim, feature_dim=256, lr=1e-3):
        super(ICMneural, self).__init__()
        _, _, c = obs_shape
        self.action_space = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #print("obs_shape: ", obs_shape)
        #print("action_dim: ", action_dim)

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

        # optimizer y losses
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # loss para modelo directo
        self.loss_fn = nn.MSELoss()
        # loss para modelo inverso
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

        state_, next_state_, pred_action, pred_next_state = self.forward(state, next_state, action)

        inv_loss = self.loss_inv(pred_action, action)
        forw_loss = self.loss_fn(pred_next_state, next_state_)
        total_loss = inv_loss + forw_loss
        
        intrinsic_reward = self.loss_fn(pred_next_state, next_state_).detach().cpu().numpy()

        return intrinsic_reward, total_loss
    
    def select_action(self, obs):

        print("obs shape:", obs.shape)

        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0) / 255.0
        obs = obs.permute(0, 3, 1, 2)

        # extraer caracteristicas
        _obs = self.feature_extractor(obs)

        max_curiosity = -float("inf")
        best_action = None

        for a in range(self.action_space):
            
            action_tensor = torch.tensor([a], device=self.device)
            action_onehot = F.one_hot(action_tensor, num_classes=self.action_space).float()

            forward_input = torch.cat([_obs, action_onehot], dim=1)
            phi_next_pred = self.forward_model(forward_input)

            curiosity = F.mse_loss(phi_next_pred, _obs, reduction='sum').item()

            if curiosity > max_curiosity:
                max_curiosity = curiosity
                best_action = a
                
        print("accion: ", best_action)

        return best_action
    
    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()