import os, sys
ROOT_DIR = '//'
SRC_DIR = ROOT_DIR + 'src'
UTILS_DIR = ROOT_DIR + 'utils'
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, UTILS_DIR)
import torch as pt
import torch.nn as nn
import numpy as np
from trainer import MyVAE
from trainer import MyLSTM
from utils.loaders import produce_embeddings, produce_outputs, produce_predicts, anomaly_loader
from torch.utils.data import DataLoader


class MyLSTMVAE(nn.Module):
    def __init__(self, data_train, data_test, code_config, in_channels: int, latent_dim: int, input_size: int, hidden_size:int, latent_size:int, num_layers:int,
                 hidden_dims = None, beta: int = 4, gamma: float = 1000., max_capacity: int = 25,
                 Capacity_max_iter: int = 1e4, loss_type: str = 'B', learning_rate=1e-4, batch_size=32, num_epochs=100) -> None:
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
        self.myvae = MyVAE(data_train, in_channels, latent_dim, hidden_dims, beta, gamma, max_capacity, Capacity_max_iter, loss_type, learning_rate, batch_size)

    def train_vae(self):
        self.myvae.train(self.num_epochs)
        self.loss_vae_train, self.input_train, self.reconst_train = self.myvae.test(self.data_train, load_model=False)
        self.loss_vae_test, self.input_test, self.reconst_test = self.myvae.test(self.data_test, load_model=False)
        return

    def embedding(self):
        self.myvae.eval()
        lstm_data_train = {}
        lstm_data_train['train_lstm'] = self.data_train
        lstm_data_train['test_lstm'] = self.data_test
        lstm_data_train['n_train_lstm'] = self.data_train.shape[0]
        lstm_data_train['n_test_lstm'] = self.data_test.shape[0]
        self.embedding_train, self.embedding_test = produce_embeddings(self.code_config, self.myvae, lstm_data_train)
        self.gt_train, self.gt_test = produce_outputs(self.code_config, lstm_data_train)
        return

    def train_lstm(self):
        self.mylstm = MyLSTM(self.embedding_train, self.input_size, self.hidden_size, self.latent_size, num_layers=self.num_layers, learning_rate=self.learning_rate, batch_size=self.batch_size)
        self.mylstm.train(self.num_epochs)
        self.loss_lstm_train, self.input_lstm_train, self.pred_lstm_train = self.mylstm.test(self.embedding_train, load_model=False)
        self.loss_lstm_test, self.input_lstm_test, self.pred_lstm_test = self.mylstm.test(self.embedding_test, load_model=False)
        return
    
    def anomaly_detect(self):
        self.reconst_pred = self.myvae.decode(self.pred_lstm_test)
        
        return
        
        
        