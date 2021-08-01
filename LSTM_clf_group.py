from gensim.models import Word2Vec
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import os
import numpy as np
import pandas as pd
import pickle
import torch.optim as optim
import torch.nn as nn
import argparse
from torch.autograd import Variable
import pdb

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

class TxtDatasetProcessing(Dataset):
    def __init__(self, dataset, seqlen, lyric_vocabulary):
        self.dataset = dataset
        lyrics = []
        musics = []
        labels = []
        for i in range(dataset.shape[0]):
            musics += list(dataset[i, 0, :])
            lyrics += list(dataset[i, 1, :])
            labels += list(dataset[i, 3, :])

        index = []
        for i in range(len(labels)):
            if labels[i] == 'negative':
                index.append(i)
        
        negative_musics = []
        negative_lyrics = []
        negative_labels = []
        for i in index:
            negative_musics.append(musics[i])
            negative_lyrics.append(lyrics[i])
            negative_labels.append(labels[i])

        self.lyrics = lyrics+negative_lyrics*2
        self.musics = musics+negative_musics*2
        self.labels = labels+negative_labels*2
        self.seqlen = seqlen
        self.lyric_vocabulary = lyric_vocabulary


    def __getitem__(self, index):
        lyric = self.lyrics[index]
        music = self.musics[index]
        la = self.labels[index]
        if la == 'positive':
            label = torch.ones(1)
        else:
            label = torch.zeros(1)

        #txt = torch.zeros((seqlen, 23), dtype = torch.float64)
        txt = torch.LongTensor(np.zeros(self.seqlen, dtype=np.int64))
        txt_len = 0
        for i in range(len(lyric)):
            for j in range(len(lyric[i])):
                syll = lyric[i][j]
                #note = music[i][j]
                if syll in lyric_vocabulary.keys():
                    syll2idx = lyric_vocabulary[syll]
                else:
                    continue
                #syllWordVec = (syll,word,note)
                if txt_len<self.seqlen:
                    txt[txt_len] = syll2idx
                    txt_len += 1
                else:
                    break
            if txt_len >= self.seqlen:
                break
        return txt, label.type(torch.int64)

    def __len__(self):
        return len(self.labels)

class LSTMClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes, lyric_vocabulary):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        vocab_size = len(lyric_vocabulary)
        self.word_embeddings = nn.Embedding(vocab_size, input_size)

    def forward(self, X):
        h0 = torch.zeros(self.num_layers*2, X.size(0), self.hidden_size).to(device) # 同样考虑向前层和向后层
        c0 = torch.zeros(self.num_layers*2, X.size(0), self.hidden_size).to(device)
        lens = len(X)
        batch_size = X.size(0)
        X = self.word_embeddings(X)
        #X = X.view(lens, batch_size, -1)

        out, _ = self.lstm(X, (h0, c0))  # LSTM输出大小为 (batch_size, seq_length, hidden_size*2)
        
        out = self.fc(out[:, -1, :])
        #out = self.Sigmoid(out)
        return out

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='CLF.py')
    parser.add_argument('--data', type=str, default='lyrics_datasets/dataset_20_group_8.npy', help="Dnd data.")
    opt = parser.parse_args()
    dataset_len = 20
    batch_size = 64
    seqlen = 30
    learning_rate = 0.0001
    num_epochs = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # save np.load
    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

    dataset = np.load(opt.data)

    lyric_vocabulary_file = 'saved_model/lyric_vocabulary_clf.npy'
    lyric_vocabulary = np.load(lyric_vocabulary_file)
    lyric_vocabulary = lyric_vocabulary.item()

    model = LSTMClassifier(input_size=128, hidden_size=256, num_layers=6, num_classes=2, lyric_vocabulary=lyric_vocabulary)
    model = model.to(device)

    for i in range(8):
        train_dataset = np.concatenate((dataset[0:i], dataset[i+1:]), axis=0)
        test_dataset = dataset[i:i+1]

        dtrain_set = TxtDatasetProcessing(train_dataset, seqlen, lyric_vocabulary)
        #weights = [3.0 if label == 'negative' else 1.0 for data, label in dtrain_set]
        #train_sampler = WeightedRandomSampler(weights, num_samples = len(weights))

        train_loader = DataLoader(dtrain_set, batch_size=batch_size, shuffle=True, num_workers=4)
        dtest_set = TxtDatasetProcessing(test_dataset, seqlen, lyric_vocabulary)
        test_loader = DataLoader(dtrain_set, batch_size=batch_size, shuffle=True, num_workers=4)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        weight = torch.FloatTensor([3,1]).to(device)
        #loss_function = nn.CrossEntropyLoss(weight = weight)
        loss_function = nn.CrossEntropyLoss()
        train_loss_ = []
        test_loss_ = []
        train_acc_ = []
        test_acc_ = []

        for epoch in range(num_epochs):
            optimizer = adjust_learning_rate(optimizer, epoch)
            total_acc = 0.0
            total_loss = 0.0
            total = 0.0
            for iter, traindata in enumerate(train_loader):
                train_inputs, train_labels = traindata
                train_labels = torch.squeeze(train_labels)
                train_inputs = Variable(train_inputs.to(device))
                train_labels = Variable(train_labels.to(device))

                model.zero_grad()
                output = model(train_inputs)
                output = output.squeeze(dim=-1)

                loss = loss_function(output, Variable(train_labels))
                loss.backward()
                optimizer.step()

                #predicted = 1*(output>0.5)
                _, predicted = torch.max(output.data, 1)
                total_acc += (predicted == train_labels).sum()
                total_acc = total_acc.type(torch.float32)
                total += torch.tensor(len(train_labels))
                total = total.type(torch.float32)
                total_loss += loss.data

            train_loss_.append(total_loss / total)
            train_acc_.append(total_acc / total)
            # testing epoch
            total_acc = 0.0
            total_loss = 0.0
            total = 0.0
            for iter, testdata in enumerate(test_loader):
                test_inputs, test_labels = testdata
                test_labels = torch.squeeze(test_labels)
                test_inputs = Variable(test_inputs.to(device))
                test_labels = Variable(test_labels.to(device))

                output = model(test_inputs)
                output = output.squeeze(dim=-1)

                loss = loss_function(output, Variable(test_labels))

                # calc testing acc
                _, predicted = torch.max(output.data, 1)
                #pdb.set_trace()
                #predicted = 1*(output>0.5)
                total_acc += (predicted == test_labels).sum()
                total_acc = total_acc.type(torch.float32)
                total += torch.tensor(len(test_labels))
                total = total.type(torch.float32)
                total_loss += loss.data

            test_loss_.append(total_loss / total)
            test_acc_.append(total_acc / total)
            print('[Epoch: %3d/%3d] Training Loss: %.3f, Testing Loss: %.3f, Training Acc: %.3f, Testing Acc: %.3f'
              % (epoch, num_epochs, train_loss_[epoch], test_loss_[epoch], train_acc_[epoch], test_acc_[epoch]))

        filename = 'binaryEmotion_datasetlen_'+str(dataset_len)+'_fold_'+str(i)+'_clf_LSTM_.pkl'
        torch.save(model.state_dict(), filename)
        print('File %s is saved.' % filename)