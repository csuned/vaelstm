import os, sys
ROOT_DIR = '../'
SRC_DIR = ROOT_DIR + 'src'
UTILS_DIR = ROOT_DIR + 'utils'
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, UTILS_DIR)
import torch as pt
import torch.nn as nn
import numpy as np
from trainer import MyVAE
from trainer import MyLSTM
from loaders import produce_embeddings, produce_outputs, produce_predicts, anomaly_loader
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pdb

class MyLSTMVAE(nn.Module):
    
    def __init__(self, data_train, data_test, code_config, total_batch, in_channels: int, latent_dim: int, input_size: int, hidden_size:int, latent_size:int, num_layers:int, hidden_dims = None, beta: int = 4, gamma: float = 1., max_capacity: int = 25,
Capacity_max_iter: int = 1e4, loss_type: str = 'B', seq_len=48, learning_rate=1e-4, batch_size=32, num_epochs=100, device='cuda0') -> None:
        super(MyLSTMVAE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.data_train = data_train
        self.data_test = data_test
        self.code_config = code_config
        self.device = device
        self.seq_len = seq_len
        self.myvae = MyVAE(data_train, in_channels, batch_size, total_batch, latent_dim, hidden_dims, beta, gamma, max_capacity, Capacity_max_iter, loss_type, learning_rate, batch_size, seq_len, device=device)

    def train_vae(self):
        self.myvae.train(self.num_epochs)
        loss_vae_train, input_train, reconst_train = self.myvae.test(self.data_train, load_model=False)
        loss_vae_test, input_test, reconst_test = self.myvae.test(self.data_test, load_model=False)
        return loss_vae_train, input_train, reconst_train, loss_vae_test, input_test, reconst_test

    def embedding(self):
        self.myvae.eval()
        lstm_data_train = {}
        lstm_data_train['train_lstm'] = self.data_train.to(self.device)
        lstm_data_train['test_lstm'] = self.data_test.to(self.device)
        lstm_data_train['n_train_lstm'] = self.data_train.shape[0]
        lstm_data_train['n_test_lstm'] = self.data_test.shape[0]
        embedding_train, embedding_test = produce_embeddings(self.code_config, self.myvae, lstm_data_train, self.device)
        self.data_train_lstm, self.data_test_lstm = produce_predicts(self.code_config, embedding_train, embedding_test, self.device)
        gt_train, gt_test = produce_outputs(self.code_config, lstm_data_train)
        return gt_train.cpu().detach().numpy(), gt_test.cpu().detach().numpy()

    def train_lstm(self):
        self.mylstm = MyLSTM(self.data_train_lstm, self.input_size, self.hidden_size, self.latent_size, self.code_config, num_layers=self.num_layers, learning_rate=self.learning_rate, batch_size=self.batch_size, device=self.device)
        self.mylstm.train(self.num_epochs)
        loss_lstm_train, input_lstm_train, pred_lstm_train = self.mylstm.test(self.data_train_lstm, load_model=False)
        loss_lstm_test, input_lstm_test, pred_lstm_test = self.mylstm.test(self.data_test_lstm, load_model=False)
        return loss_lstm_train, input_lstm_train, pred_lstm_train, loss_lstm_test, input_lstm_test, pred_lstm_test
    
    def anomaly_detect(self, pred_lstm): #input the prediction of LSTM and ground truth ouput of decoder from here.
        pred_lstm = pred_lstm.reshape(pred_lstm.shape[0], -1) #reshape to hidden state
        pred = Variable(pt.tensor(pred_lstm)).to(self.device)
        reconst = self.myvae.decode(pred)
        return reconst.cpu().detach().numpy()
        
        
        