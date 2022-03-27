# -*- coding: utf-8 -*-
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
    lr = opt.learning_rate * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

class TxtDatasetProcessing(Dataset):
    def __init__(self, dataset, seqlen, syllable_vocabulary, music_vocabulary):
        self.dataset = dataset
        lyrics = []
        musics = []
        labels = []
        for i in range(dataset.shape[0]):
            musics += list(dataset[i, 0, :])
            lyrics += list(dataset[i, 1, :])
            labels += list(dataset[i, 2, :]) 

        #把negative samples记录下来，多两倍
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

        self.lyrics = lyrics+negative_lyrics
        self.musics = musics+negative_musics
        self.labels = labels+negative_labels
        
        self.seqlen = seqlen
        self.word_vocabulary = word_vocabulary
        self.syllable_vocabulary = syllable_vocabulary
        self.music_vocabulary = music_vocabulary

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
        mus = torch.LongTensor(np.zeros(self.seqlen, dtype=np.int64))
        txt_len = 0
        for i in range(len(lyric)):
            word = ''
            for syll in lyric[i]:
                word += syll
            if word in self.word_vocabulary:
                word2idx = self.word_vocabulary[word]
            else:
            	continue
            for j in range(len(lyric[i])):
                syll = lyric[i][j]
                #note = music[i][j]
                if syll in self.syllable_vocabulary:
                    syll2idx = self.syllable_vocabulary[syll]
                else:
                    continue
                #syllWordVec = (syll,word,note)
                music_note = 'p_'+str(music[i][j][0])+'^'+'d_'+str(music[i][j][1])+'^'+'r_'+str(music[i][j][2])
                if music_note in self.music_vocabulary:
                    music_note2idx = self.music_vocabulary[music_note]
                else:
                    continue
                syllWordVec = (syll2idx,music_note2idx)
                if txt_len<self.seqlen:
                    txt[txt_len] = syll2idx
                    mus[txt_len] = music_note2idx
                    txt_len += 1
                else:
                    break
            if txt_len >= self.seqlen:
                break
        return txt, mus, label.type(torch.int64)

    def __len__(self):
        return len(self.labels)

class LSTMClassifier(nn.Module):

    def __init__(self, input_txt_size, input_mus_size, hidden_size, num_layers, num_classes, syllable_vocabulary, music_vocabulary):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        input_size = input_txt_size+input_mus_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        syllable_vocab_size = len(syllable_vocabulary)
        self.word_embeddings = nn.Embedding(syllable_vocab_size, input_txt_size)
        music_vocab_size = len(music_vocabulary)
        self.music_embeddings = nn.Embedding(music_vocab_size, input_mus_size)

    def forward(self, txt, mus):
        h0 = torch.zeros(self.num_layers*2, txt.size(0), self.hidden_size).cuda() # 同样考虑向前层和向后层
        c0 = torch.zeros(self.num_layers*2, txt.size(0), self.hidden_size).cuda()
        lens = len(txt)
        batch_size = txt.size(0)
        txt = self.word_embeddings(txt)
        mus = self.music_embeddings(mus)
        X = torch.cat((txt, mus), 2)
        #X = X.view(lens, batch_size, -1)

        out, _ = self.lstm(X, (h0, c0))  # LSTM输出大小为 (batch_size, seq_length, hidden_size*2)
        out = self.fc(out[:, -1, :])
        #out = self.Sigmoid(out)
        return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLF.py')
    parser.add_argument('--data', type=str, default='lyrics_datasets_v3/dataset_50_v3_clf.npy', help="Dnd data.")
    parser.add_argument('--dataset_len', type=str, default=50)
    parser.add_argument('--batch_size', type=str, default=32)
    parser.add_argument('--seqlen', type=str, default=70)
    parser.add_argument('--learning_rate', type=str, default=0.0001)
    parser.add_argument('--num_epochs', type=str, default=30)
    opt = parser.parse_args()
    #dataset_len = 20
    #batch_size = 64
    #seqlen = 30
    #learning_rate = 0.0001
    #num_epochs = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # save np.load
    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

    dataset = np.load(opt.data) #shape (8, 4, 4722)

    word_vocabulary_file = 'saved_model/word_vocabulary.npy'
    word_vocabulary = np.load(word_vocabulary_file)
    word_vocabulary = word_vocabulary.item()
    syllable_vocabulary_file = 'saved_model/syllable_vocabulary.npy'
    syllable_vocabulary = np.load(syllable_vocabulary_file)
    syllable_vocabulary = syllable_vocabulary.item()
    music_vocabulary_file = 'saved_model/music_vocabulary_'+str(opt.dataset_len)+'.npy'
    music_vocabulary = np.load(music_vocabulary_file)
    music_vocabulary = music_vocabulary.item()

    model = LSTMClassifier(input_txt_size=128, input_mus_size=10, hidden_size=256, num_layers=6, num_classes=2, syllable_vocabulary=syllable_vocabulary, music_vocabulary=music_vocabulary)
    model = model.to(device)

    for i in range(8):
        train_dataset = np.concatenate((dataset[0:i], dataset[i+1:]), axis=0)
        test_dataset = dataset[i:i+1]

        dtrain_set = TxtDatasetProcessing(train_dataset, opt.seqlen, syllable_vocabulary, music_vocabulary)
        train_loader = DataLoader(dtrain_set, batch_size=opt.batch_size, shuffle=True, num_workers=4)
        
        # data[0] torch.Size([64, 30])  data[1] torch.Size([64, 30])   data[2] torch.Size([64, 1])
        #for data in train_loader:
        #    print(data)

        dtest_set = TxtDatasetProcessing(test_dataset, opt.seqlen, syllable_vocabulary, music_vocabulary)
        test_loader = DataLoader(dtrain_set, batch_size=opt.batch_size, shuffle=True, num_workers=4)

        optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        train_loss_ = []
        test_loss_ = []
        train_acc_ = []
        test_acc_ = []

        for epoch in range(opt.num_epochs):
            optimizer = adjust_learning_rate(optimizer, epoch)
            total_acc = 0.0
            total_loss = 0.0
            total = 0.0
            for iter, traindata in enumerate(train_loader):
                train_txt, train_mus ,train_labels = traindata
                train_labels = torch.squeeze(train_labels)
                train_txt = Variable(train_txt.to(device))
                train_mus = Variable(train_mus.to(device))
                train_labels = Variable(train_labels.to(device))

                model.zero_grad()
                output = model(train_txt, train_mus)
                output = output.squeeze(dim=-1)

                loss = loss_function(output, Variable(train_labels))
                loss.backward()
                optimizer.step()

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
                test_txt, test_mus, test_labels = testdata
                test_labels = torch.squeeze(test_labels)
                test_txt = Variable(test_txt.to(device))
                test_mus = Variable(test_mus.to(device))
                test_labels = Variable(test_labels.to(device))

                output = model(test_txt, test_mus)
                output = output.squeeze(dim=-1)

                loss = loss_function(output, Variable(test_labels))

                # calc testing acc
                _, predicted = torch.max(output.data, 1)

                #predicted = 1*(output>0.5)
                total_acc += (predicted == test_labels).sum()
                total_acc = total_acc.type(torch.float32)
                total += torch.tensor(len(test_labels))
                total = total.type(torch.float32)
                total_loss += loss.data

            test_loss_.append(total_loss / total)
            test_acc_.append(total_acc / total)
            print('[Epoch: %3d/%3d] Training Loss: %.3f, Testing Loss: %.3f, Training Acc: %.3f, Testing Acc: %.3f'
              % (epoch, opt.num_epochs, train_loss_[epoch], test_loss_[epoch], train_acc_[epoch], test_acc_[epoch]))

        filename = 'LSTM_datasetlen_'+str(opt.dataset_len)+'_fold_'+str(i)+'_clf.pkl'
        torch.save(model.state_dict(), filename)
        print('File %s is saved.' % filename)