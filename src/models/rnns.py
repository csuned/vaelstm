import torch as pt
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable
import pdb

class denselstm(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers=1, code_config=None):
        super(denselstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*code_config['num_in']*code_config['hidden_len'], hidden_size*code_config['hidden_len'])

    def forward(self, x, device):
        h_0 = Variable(pt.tanh(pt.rand(self.num_layers, x.size(0), self.hidden_size))).to(device)
        c_0 = Variable(pt.tanh(pt.rand(self.num_layers, x.size(0), self.hidden_size))).to(device)
        x = pt.squeeze(x, 3)
        x = x.permute(0,2,1)     
        out, _ = self.lstm(x, (h_0, c_0))
        out = out.permute(0,2,1)
        out = self.fc(out.reshape(out.size(0),-1))
        return out.reshape(out.size(0), self.latent_size, -1, 1)

