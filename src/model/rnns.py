import torch as pt
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable

class denselstm(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers=1):
        super(denselstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

