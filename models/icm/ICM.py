import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ICMneural(nn.Module):
    def __init__(self, obs_shape, action_dim, lr=1e-3):
        super(ICMneural, self).__init__()
        _, _, c = obs_shape
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # f.e. del paper "Curiosity-driven Exploration by Self-supervised Prediction"
        self.feature_extractor = nn.Sequential(

            nn.AvgPool2d(2, 2), # reducir a 42, 42 dado que recibe 84, 84
            nn.Conv2d(c, 32, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten()

        )

        # modelo inverso, predice la accion
        self.inverse_model = nn.Sequential(

            nn.Linear(288 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim)

        )

        # modelo directo, predice el siguiente estado
        self.forward_model = nn.Sequential(

            nn.Linear(288 + self.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 288)
            
        )

        # optimizer y losses
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # loss para modelo directo
        self.loss_fn = nn.MSELoss()
        # loss para modelo inverso
        self.loss_inv = nn.CrossEntropyLoss()

        self.to(self.device)

    def forward(self, state, next_state, action):

        # obtiene las caracteristicas de las observaciones
        state_ = self.feature_extractor(state)
        next_state_ = self.feature_extractor(next_state)

        pred_action = self.inverse_model(torch.cat((state_, next_state_), dim=1))

        action_onehot = F.one_hot(action, num_classes=self.action_dim).float()
        forward_input = torch.cat((state_, action_onehot), dim=1)
        pred_next_state = self.forward_model(forward_input)


        return state_, next_state_, pred_action, pred_next_state


    def get_intrinsic_reward(self, state, next_state, action):

        # convertir a tensores y normalizar
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0) / 255.0
        next_state = torch.FloatTensor(next_state).to(self.device).unsqueeze(0) / 255.0
        
        state = state.permute(0, 3, 1, 2)
        next_state = next_state.permute(0, 3, 1, 2)
        action = torch.LongTensor([action]).to(self.device)

        state_, next_state_, pred_action, pred_next_state = self.forward(state, next_state, action)

        # calculo de loss
        inv_loss = self.loss_inv(pred_action, action)
        forw_loss = self.loss_fn(pred_next_state, next_state_)
        total_loss = inv_loss + forw_loss
        
        intrinsic_reward = self.loss_fn(pred_next_state, next_state_).detach().cpu().numpy()

        return intrinsic_reward, total_loss
    
    def select_action(self, obs):

        # convertir a tensor y normalizar
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0) / 255.0
        obs = obs.permute(0, 3, 1, 2) # (1, 1, 84, 84)

        # extraer caracteristicas
        _obs = self.feature_extractor(obs) #(1, 256)

        # repetir la observacion para cada accion
        _obs_n = _obs.repeat(self.action_dim, 1) # (7, 256)
        
        action_vector = torch.eye(self.action_dim).to(self.device) # (7,7)

        # concatenar las observaciones con las acciones
        forward_input = torch.cat((_obs_n, action_vector), dim=1) # (7, 263)
        

        obs_next = self.forward_model(forward_input) # (7, 256)

        curiosity = F.mse_loss(obs_next, _obs_n, reduction='none').mean(dim=1)
        
        action = torch.argmax(curiosity).item()
        #print(action)

        return action


    def update(self, loss):

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()