import numpy as np
import torch as pt
import csv
from torch.utils.data import Dataset
from torch.autograd import Variable

def vae_training_data(batch_len, csv_file=None, in_channel=2):
    trainingdata={}
    if csv_file==None:
        print('****Using Fake Training Data with Normal Distribution****')
        trainingdata=pt.abs(pt.randn(2000, in_channel, batch_len, 1))
        trainingdata=Variable(trainingdata/pt.max(trainingdata))
    else:
        train_in = []
        train_out = []
        with open(csv_file) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=",")
            header = next(csv_reader)
            for row in csv_reader:
                train_in += [float(row[1]),]
        total_len = int((len(train_in)//batch_len)*batch_len)
        trainingdata = Variable(pt.FloatTensor(train_in[:total_len]).reshape(-1, 1, batch_len, 1))
    return trainingdata, trainingdata.size(0)

def produce_embeddings(config, model_vae, data, device):
    embedding_lstm_train = Variable(pt.zeros((data['n_train_lstm'], config['code_size'], config['hidden_len'], 1))).to(device)
    for i in range(data['n_train_lstm']):
        hidden_para = model_vae.b_vae.encode(pt.unsqueeze(data['train_lstm'][i],0))
        embedding_lstm_train[i] = model_vae.b_vae.reparameterize(hidden_para[0], hidden_para[1]).view(-1, 256, model_vae.b_vae.hidden_len, 1)

    embedding_lstm_test = Variable(pt.zeros((data['n_test_lstm'], config['code_size'], config['hidden_len'], 1))).to(device)
    for i in range(data['n_test_lstm']):
        hidden_para = model_vae.b_vae.encode(pt.unsqueeze(data['test_lstm'][i],0))
        embedding_lstm_test[i] = model_vae.b_vae.reparameterize(hidden_para[0], hidden_para[1]).view(-1, 256, model_vae.b_vae.hidden_len, 1)
    return embedding_lstm_train, embedding_lstm_test

def produce_predicts(config, embedding_lstm_train, embedding_lstm_test, device):#get lstm training data from VAE latents
    n_train = embedding_lstm_train.shape[0]
    n_test = embedding_lstm_test.shape[0]
    lstm_train_in = Variable(pt.zeros((n_train-config['num_in'], config['code_size'], config['hidden_len']*config['num_in'], 1))).to(device)
    lstm_test_in = Variable(pt.zeros((n_test-config['num_in'], config['code_size'], config['hidden_len']*config['num_in'], 1))).to(device)
    lstm_train_out = Variable(pt.zeros((n_train-config['num_in'], config['code_size'], config['hidden_len'], 1))).to(device)
    lstm_test_out = Variable(pt.zeros((n_test-config['num_in'], config['code_size'], config['hidden_len'], 1))).to(device)
    for i in range(n_train):
        if i+config['num_in']<n_train:
            for n in range(config['num_in']):
                lstm_train_in[i, :, config['hidden_len']*n:config['hidden_len']*(n+1)] = embedding_lstm_train[i+n]
            lstm_train_out[i] = embedding_lstm_train[i+config['num_in']]
    for i in range(n_test):
        if i+config['num_in']<n_test:
            for n in range(config['num_in']):
                lstm_test_in[i, :, config['hidden_len']*n:config['hidden_len']*(n+1)] = embedding_lstm_test[i+n]
            lstm_test_out[i] = embedding_lstm_test[i+config['num_in']]
    data_train_lstm = {}
    data_test_lstm = {}
    data_train_lstm['input'] = lstm_train_in
    data_train_lstm['output'] = lstm_train_out
    data_test_lstm['input'] = lstm_test_in
    data_test_lstm['output'] = lstm_test_out
    return data_train_lstm, data_test_lstm

def produce_outputs(config, data): #output of the decoder
    eval_train_out = data['train_lstm'][config['num_in']:,:,:,:]
    eval_test_out = data['test_lstm'][config['num_in']:,:,:,:]
    return eval_train_out, eval_test_out

class anomaly_loader(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.out_dim = len(dataset.shape)

    def __len__(self):
        return int(self.dataset.shape[0])

    def __getitem__(self, idx):
        return self.dataset[idx,:]

class embedding_loader(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.out_dim = len(dataset['output'].shape)

    def __len__(self):
        return int(self.dataset['input'].shape[0])

    def __getitem__(self, idx):
        item={}
        item['input'] = self.dataset['input'][idx,:,:]
        item['output'] = self.dataset['output'][idx,:,:]
        return item