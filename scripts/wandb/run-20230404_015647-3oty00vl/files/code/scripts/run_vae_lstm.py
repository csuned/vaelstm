import os, sys
ROOT_DIR = '/home/jupyter-chuanhao/data/chuanhao_anomaly/vaelstm/'
SRC_DIR = ROOT_DIR + 'src'
UTILS_DIR = ROOT_DIR + 'utils'
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, UTILS_DIR)
from vaelstm import MyLSTMVAE
from myplot import plt_compare_series
from loaders import vae_training_data
import torch as pt
import wandb
import pdb


'''
Define Global Const
code_config: per code length, number of code used as input for prediction, dimension of code (2 for b_vae)
data_train, data_test: training set and evaluation set, add validation set if you want
in_channels, latent_dim, hidden_dims, beta, gamma, max_capacity, Capacity_max_iter, loss_type: hyper parameters of VAE
input_size, hidden_size, latent_size, num_layers: hyper parameters of LSTM
learning_rate=1e-4, batch_size=32, num_epochs=100, device='cuda0': default traning configs
'''
code_config = {
    'l_seq':144, # this field should be aligned with your VAE latent length
    'code_size':256, # 2 for the VAE of this template
    'num_in':3, # how many embeddings to predict next embedding
    'hidden_len':9,
    }

in_channels=2
input_size=256
hidden_size=256
latent_size=input_size #LSTM does not change the dimension
num_layers=1
latent_dim=2
hidden_dims=None #using default value
beta = 4 
gamma = 1000.
max_capacity = 25
Capacity_max_iter = 1e4
loss_type = 'B'
seq_len = code_config['l_seq']
learning_rate = 1e-3
batch_size=32
#total_batch #need this to compute KLD weight
num_epochs=30
gpu_id = 0 
device = pt.device(f"cuda:{gpu_id}" if (pt.cuda.is_available() and gpu_id in [0,1]) else "cpu")


wandb.login()
run = wandb.init(
# Set the project where this run will be logged
project="MyVAELSTM",
# Track hyperparameters and run metadata
config={
    "learning_rate": learning_rate,
    "epochs": num_epochs,
    'VAE_loss_type': 'B',
})

'''
Prepare training/eval data
'''
#debuging with random input
data_train, total_batch = vae_training_data(seq_len, in_channel=in_channels)
data_test, _ = vae_training_data(seq_len, in_channel=in_channels)

'''
Model Training and Evaluation
'''
detector = MyLSTMVAE(data_train, data_test, code_config, total_batch, in_channels, latent_dim, input_size, hidden_size, latent_size, num_layers, hidden_dims, beta, gamma, max_capacity, Capacity_max_iter, loss_type, seq_len, learning_rate, batch_size, num_epochs, device).to(device)
#train VAE
loss_vae_train, input_train, reconst_train, loss_vae_test, input_test, reconst_test = detector.train_vae()
#train LSTM
gt_train, gt_test = detector.embedding()
loss_lstm_train, input_lstm_train, pred_lstm_train, loss_lstm_test, input_lstm_test, pred_lstm_test = detector.train_lstm()
#Carry out anomaly detection
reconst = detector.anomaly_detect(pred_lstm_test)

'''
Plot, local result see results (only save latest result)
'''
plt_compare_series([gt_test[:3], reconst[:3]], 'Predictions', label_list=['Ground Truth', 'Predictions'])

run.finish()



