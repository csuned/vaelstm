import numpy as np
import torch as pt
import csv
from torch.utils.data import Dataset
from torch.autograd import Variable

def produce_embeddings(config, model_vae, data):
    embedding_lstm_train = np.zeros((data['n_train_lstm'], config['l_seq'], config['code_size']))
    for i in range(data['n_train_lstm']):
        embedding_lstm_train[i] = model_vae.encode(data['train_lstm'][i])

    embedding_lstm_test = np.zeros((data['n_test_lstm'], config['l_seq'], config['code_size']))
    for i in range(data['n_test_lstm']):
        embedding_lstm_test[i] = model_vae.encode(data['test_lstm'][i])
    return embedding_lstm_train, embedding_lstm_test

def produce_predicts(config, embedding_lstm_train, embedding_lstm_test):
    n_train = embedding_lstm_train.shape[0]
    n_test = embedding_lstm_test.shape[0]
    lstm_train_in = np.zeros((n_train-config['num_in'], config['l_seq']*config['num_in'], config['code_size']))
    lstm_test_in = np.zeros((n_test-config['num_in'], config['l_seq']*config['num_in'], config['code_size']))
    lstm_train_out = np.zeros((n_train-config['num_in'], config['l_seq'], config['code_size']))
    lstm_test_out = np.zeros((n_test-config['num_in'], config['l_seq'], config['code_size']))
    for i in range(n_train):
        if i+config['num_in']<n_train:
            for n in range(config['num_in']):
                lstm_train_in[i, config['l_seq']*n:config['l_seq']*(n+1)] = embedding_lstm_train[i+n]
            lstm_train_out[i] = embedding_lstm_train[i+config['num_in']]
    for i in range(n_test):
        if i+config['num_in']<n_test:
            for n in range(config['num_in']):
                lstm_test_in[i, config['l_seq']*n:config['l_seq']*(n+1)] = embedding_lstm_test[i+n]
            lstm_test_out[i] = embedding_lstm_test[i+config['num_in']]
    return lstm_train_in, lstm_test_in, lstm_train_out, lstm_test_out

def produce_outputs(config, data): #output of the decoder
    batch_len = data['train_lstm'].shape[1]
    channel_num = data['train_lstm'].shape[2]
    eval_train_out = np.zeros((data['n_train_lstm']-config['num_in'], batch_len, channel_num))
    eval_test_out = np.zeros((data['n_test_lstm']-config['num_in'], batch_len, channel_num))
    eval_train_out = data['train_lstm'][config['num_in']:]
    eval_test_out = data['test_lstm'][config['num_in']:]
    return eval_train_out, eval_test_out

def vae_training_data(batch_len, csv_file=None):
    trainingdata={}
    if csv_file==None:
        print('****Using Fake Training Data with Normal Distribution****')
        trainingdata['input']=Variable(pt.randn(2e3, batch_len,1))
        trainingdata['output'] = Variable(pt.randn(2e3, batch_len, 1))
    else:
        train_in = []
        train_out = []
        with open(csv_file) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=",")
            header = next(csv_reader)
            for row in csv_reader:
                train_in += [float(row[1]),]
                train_out += [float(row[1]), ]
        total_len = int((len(train_in)//batch_len)*batch_len)
        trainingdata['input'] = Variable(pt.FloatTensor(train_in[:total_len]).reshape(-1, batch_len, 1))
        trainingdata['output'] = Variable(pt.FloatTensor(train_out[:total_len]).reshape(-1, batch_len, 1))
    return trainingdata

class anomaly_loader(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.out_dim = len(dataset['output'].shape)

    def __len__(self):
        return int(self.dataset['input'].shape[0])

    def __getitem__(self, idx):
        item={}
        item['input'] = self.dataset['input'][idx,:]
        item['output'] = self.dataset['output'][idx,:]
        return item

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