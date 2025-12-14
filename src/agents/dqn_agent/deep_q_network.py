import torch


class DeepQNetwork(torch.nn.Module):
    def __init__(self, input_size, output_size, r_dropout=0.2):
        super().__init__()
        self.sequence = torch.nn.Sequential(
            torch.nn.Linear(input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
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
            torch.nn.Linear(256, output_size),
        )

    def forward(self, input_tensor):
        output_tensor = self.sequence(input_tensor)
        return output_tensor
