import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()

        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        # feed input through nn
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)
        return output

class FeedForwardImageNetwork(nn.Module):
    def __init__(self, out_dim):
        super(FeedForwardNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))

        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(p=0.2)

        self.action_value1 = nn.Linear(3136, 1024)
        self.action_value2 = nn.Linear(1024, 1024)
        self.action_value3 = nn.Linear(1024, out_dim)

    """
    Push an observation through the neural network
    """
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        # feed input through nn
        obs1 = self.relu(self.conv1(obs))
        obs2 = self.relu(self.conv2(obs1))
        obs3 = self.relu(self.conv3(obs2))
        obs_flattened = self.flatten(obs3)
        action_value1 = self.relu(self.action_value1(obs_flattened))
        action_value2 = self.relu(self.action_value2(action_value1))
        output = self.action_value3(action_value2)
        return output

    def save_the_model(self, weights_filename="models/latest.pt"):
        if not os.path.exists("models"):
            os.makedirs("models")
        torch.save(self.state_dict(), weights_filename)

    def load_the_model(self, weights_filename="models/latest.pt"):
        try:
            self.load_state_dict(torch.load(weights_filename))
            print(f'Sucessfully loaded model from {weights_filename}')
        except:
            print(f'No weights file available at {weights_filename}')


