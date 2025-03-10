import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FUNCTION_CONFIG
import torch
import torch.nn as nn
import torch.nn.functional as F

class RegretNetwork(nn.Module):
    def __init__(self, hidden_sizes=[128, 64, 32]):
        super().__init__()
        input_size = FUNCTION_CONFIG["obs_dim"] + 1
        layers = [nn.Linear(input_size, hidden_sizes[0]), nn.ReLU()]
        
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x):
        x = self.network(x)  
        output = self.output_layer(x)  
        return torch.tanh(output).squeeze(-1)  # (batch,)


class PolicyNetwork(nn.Module):
    def __init__(self, hidden_sizes=[256, 256, 128]):
        super().__init__()
        input_size = FUNCTION_CONFIG["obs_dim"]
        layers = [nn.Linear(input_size, hidden_sizes[0]), nn.ReLU()]
        
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_sizes[-1], FUNCTION_CONFIG["n_discrete"])

    def forward(self, x):
        x = self.network(x)  
        log_probs = torch.log_softmax(self.output_layer(x), dim=-1)  # Log-probabilities
        return log_probs
