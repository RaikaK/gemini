import torch


class Actor(torch.nn.Module):
    def __init__(self, input_size, actor_output_size=1, r_dropout=0.2):
        super().__init__()
        self.sequence = torch.nn.Sequential(
            torch.nn.Linear(input_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(r_dropout),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(r_dropout),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(r_dropout),
        )
        self.latent = torch.nn.Linear(256, actor_output_size)

    def forward(self, input_tensor):
        features = self.sequence(input_tensor)
        action_value = self.latent(features)
        return action_value


class Critic(torch.nn.Module):
    def __init__(self, input_size, r_dropout=0.2):
        super().__init__()
        self.sequence = torch.nn.Sequential(
            torch.nn.Linear(input_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(r_dropout),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(r_dropout),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(r_dropout),
        )
        self.latent = torch.nn.Linear(256, 1)

    def forward(self, input_tensor):
        features = self.sequence(input_tensor)
        state_value = self.latent(features)
        return state_value
