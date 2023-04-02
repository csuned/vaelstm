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

    return lstm_train_out, lstm_test_out

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