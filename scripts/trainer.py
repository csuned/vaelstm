import os, sys
ROOT_DIR = '//'
SRC_DIR = ROOT_DIR + 'src'
UTILS_DIR = ROOT_DIR + 'utils'
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, UTILS_DIR)
import torch as pt
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable
from src.model.vae import BetaVAE
from src.model.rnns import denselstm
from utils.loaders import anomaly_loader, embedding_loader
from torch.utils.data import DataLoader
class MyVAE(nn.Module):

    def __init__(self, data_train, in_channels: int, latent_dim: int, hidden_dims = None, beta: int = 4, gamma: float = 1000., max_capacity: int = 25,
                 Capacity_max_iter: int = 1e4, loss_type: str = 'B', learning_rate=1e-4, batch_size=32) -> None:
        super(MyVAE, self).__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.max_capacity = max_capacity
        self.C_stop_iter = Capacity_max_iter
        self.loss_type = loss_type
        self.learning_rate = learning_rate

        self.b_vae = BetaVAE(in_channels, latent_dim, hidden_dims, beta, gamma, max_capacity, Capacity_max_iter, loss_type)

        self.device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
        self.optimizer = pt.optim.Adam(self.b_vae.parameters(), lr=learning_rate, betas=(0.5, 0.999), weight_decay=0)
        loader = anomaly_loader(data_train)
        self.dataloader = DataLoader(loader, batch_size=batch_size, shuffle=True, drop_last=True)
        print(self.b_vae)

    def train(self, num_epochs=100):
        self.b_vae.to(self.device)
        for epoch in range(num_epochs):
            for i, (x, _) in enumerate(self.dataloader):
                x = x.to(self.device)
                self.optimizer.zero_grad()
                recon_x, _, mu, logvar = self.b_vae(x)
                loss = self.b_vae.loss_function(recon_x, x, mu, logvar)
                loss['loss'].backward()
                self.optimizer.step()
                if i % 10 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, num_epochs, i + 1, len(self.dataloader), loss.item()))
        pt.save(self.b_vae.state_dict(), 'vae.pth')
        return self.b_vae

    def test(self, data_test, load_model=False):
        if load_model:
            self.b_vae.load_state_dict(pt.load('vae.pth'))
            self.b_vae.to(self.device)
        self.b_vae.eval()
        loader = anomaly_loader(data_test)
        eval_dataloader = DataLoader(loader, batch_size=1, shuffle=True, drop_last=True)
        for i, (x, _) in enumerate(eval_dataloader):
            x = x.to(self.device)
            recon_x, _, mu, logvar = self.b_vae(x)
            loss = self.b_vae.loss_function(recon_x, x, mu, logvar)
            if i>0:
                out_loss = pt.cat((out_loss, loss['loss']), 0)
                out_x = pt.cat((out_x, x), 0)
                out_recon_x = pt.cat((out_recon_x, recon_x), 0)
            else:
                out_loss = loss['loss']
                out_x = x
                out_recon_x = recon_x
        return out_loss.cpu().detach().numpy(), out_x.cpu().detach().numpy(), out_recon_x.cpu().detach().numpy()

    def load_model(self, path):
        self.b_vae.load_state_dict(pt.load(path))
        self.b_vae.to(self.device)
        self.b_vae.eval()
        return
    def encode(self, x):
        return self.b_vae.encode(x)

    def decode(self, z):
        return self.b_vae.decode(z)

class MyLSTM(nn.Module):
    def __init__(self, data_train, input_size, hidden_size, latent_size, num_layers=1, learning_rate=1e-4, batch_size=32):
        super(MyLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.prednet = denselstm(input_size, hidden_size, num_layers) #prednet: prediction network
        self.device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
        self.optimizer = pt.optim.Adam(self.prednet.parameters(), lr=learning_rate, betas=(0.5, 0.999), weight_decay=0)
        loader = embedding_loader(data_train)
        self.dataloader = DataLoader(loader, batch_size=batch_size, shuffle=True, drop_last=True)
        self.criteria = nn.L1Loss()
        print(self.prednet)

    def train(self, num_epochs=100):
        self.prednet.to(self.device)
        for epoch in range(num_epochs):
            for i, (x, _) in enumerate(self.dataloader):
                x = x.to(self.device)
                self.optimizer.zero_grad()
                y = self.prednet(x)
                loss = self.criteria(y, x)
                loss.backward()
                self.optimizer.step()
                if i % 10 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, num_epochs, i + 1, len(self.dataloader), loss.item()))
        pt.save(self.prednet.state_dict(), 'prednet.pth')
        return self.prednet

    def test(self, data_test, load_model=False):
        if load_model:
            self.prednet.load_state_dict(pt.load('prednet.pth'))
            self.prednet.to(self.device)
        self.prednet.eval()
        loader = anomaly_loader(data_test)
        eval_dataloader = DataLoader(loader, batch_size=1, shuffle=True, drop_last=True)
        for i, (x, _) in enumerate(eval_dataloader):
            x = x.to(self.device)
            y = self.prednet(x)
            loss = self.criteria(y, x)
            if i>0:
                out_loss = pt.cat((out_loss, loss), 0)
                out_x = pt.cat((out_x, x), 0)
                out_y = pt.cat((out_y, y), 0)
            else:
                out_loss = loss
                out_x = x
                out_y = y
        return out_loss.cpu().detach().numpy(), out_x.cpu().detach().numpy(), out_y.cpu().detach().numpy()

    def load_model(self, path):
        self.prednet.load_state_dict(pt.load(path))
        self.prednet.to(self.device)
        self.prednet.eval()
        return

    def predict(self, x):
        return self.prednet(x)