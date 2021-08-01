from gensim.models import Word2Vec
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch
import os
import numpy as np
import pandas as pd
import pickle
import torch.optim as optim
import torch.nn as nn
import argparse
from torch.autograd import Variable
from collections import Counter
import math
import torch.nn.functional as F
from model import Encoder, Decoder, Seq2Seq
import pdb

def data_process(lyric, music, seqlen, lyc2vec, music_vocabulary):
    lyric_input = torch.zeros((seqlen-1, lyc2vec*2), dtype = torch.float64)
    lyric_label = torch.LongTensor(np.zeros(seqlen-1, dtype=np.int64))

    music_input = torch.LongTensor(np.zeros(seqlen-1, dtype=np.int64))
    music_label = torch.LongTensor(np.zeros(seqlen-1, dtype=np.int64))
    txt_len = 0
    for i in range(len(lyric)):
        word = ''
        for syll in lyric[i]:
            word += syll
        if word in wordModel.wv.index_to_key:
            word2Vec = wordModel.wv[word]
        else:
            continue
        for j in range(len(lyric[i])):
            syll = lyric[i][j]
            note = 'p_'+str(music[i][j][0])+'^'+'d_'+str(music[i][j][1])+'^'+'r_'+str(music[i][j][2])
            note2idx = music_vocabulary[note]
            if syll in syllModel.wv.index_to_key:
                syll2Vec = syllModel.wv[syll]
                syll2idx = syllModel.wv.key_to_index[syll]
            else:
                continue
            syllWordVec = np.concatenate((word2Vec,syll2Vec))
            if txt_len<seqlen-1:
                lyric_input[txt_len] = torch.from_numpy(syllWordVec)
                music_input[txt_len] = note2idx
            if txt_len<seqlen and txt_len>0:
                lyric_label[txt_len-1] = syll2idx
                music_label[txt_len-1] = note2idx
            txt_len += 1

        if txt_len >= seqlen:
            break
        if txt_len >= seqlen:
            break
    return lyric_input.type(torch.float32), lyric_label.type(torch.int64), music_input.type(torch.int64), music_label.type(torch.int64)


def adjust_learning_rate(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def build_vocabulary(dataset):
    melodys = list(dataset[0])
    note_list = []
    for melody in melodys:
        for i in melody:
            for j in i:
                note = 'p_'+str(j[0])+'^'+'d_'+str(j[1])+'^'+'r_'+str(j[2])
                note_list.append(note)
    count = Counter(note_list)
    vocabulary = {}
    i = 0
    for key in count.keys():
    	vocabulary[key] = i
    	i+=1
    return vocabulary

class TxtDatasetProcessing(Dataset):
    def __init__(self, dataset, syllModel, wordModel, seqlen, lyc2vec, music_vocabulary):
        self.dataset = dataset
        self.lyrics = list(dataset[1])
        self.musics = list(dataset[0])
        self.syllModel = syllModel
        self.wordModel = wordModel
        self.seqlen = seqlen
        self.lyc2vec = lyc2vec
        self.music_vocabulary = music_vocabulary


    def __getitem__(self, index):
        lyric = self.lyrics[index]
        music = self.musics[index]

        lyric_input = torch.zeros((self.seqlen-1, self.lyc2vec*2), dtype = torch.float64)
        lyric_label = torch.LongTensor(np.zeros(self.seqlen-1, dtype=np.int64))

        music_input = torch.LongTensor(np.zeros(self.seqlen-1, dtype=np.int64))
        music_label = torch.LongTensor(np.zeros(self.seqlen-1, dtype=np.int64))
        txt_len = 0
        for i in range(len(lyric)):
            word = ''
            for syll in lyric[i]:
                word += syll
            if word in wordModel.wv.index_to_key:
                word2Vec = wordModel.wv[word]
            else:
                continue
            for j in range(len(lyric[i])):
                syll = lyric[i][j]
                note = 'p_'+str(music[i][j][0])+'^'+'d_'+str(music[i][j][1])+'^'+'r_'+str(music[i][j][2])
                note2idx = self.music_vocabulary[note]
                if syll in syllModel.wv.index_to_key:
                    syll2Vec = syllModel.wv[syll]
                    syll2idx = syllModel.wv.key_to_index[syll]
                else:
                    continue
                syllWordVec = np.concatenate((word2Vec,syll2Vec))
                if txt_len<self.seqlen-1:
                    lyric_input[txt_len] = torch.from_numpy(syllWordVec)
                    music_input[txt_len] = note2idx
                if txt_len<self.seqlen and txt_len>0:
                    lyric_label[txt_len-1] = syll2idx
                    music_label[txt_len-1] = note2idx
                txt_len += 1

            if txt_len >= self.seqlen:
                break
            if txt_len >= self.seqlen:
                break
        return lyric_input.type(torch.float32), lyric_label.type(torch.int64), music_input.type(torch.int64), music_label.type(torch.int64)

    def __len__(self):
        return len(self.lyrics)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='lyrics_melody_generator.py')
    parser.add_argument('--data', type=str, default='lyrics_datasets/dataset_20_group.npy', help="Dnd data.")
    parser.add_argument('--batch_size', type=str, default=64, help="batch size")
    parser.add_argument('--seqlen', type=str, default=20, help="seqlen")
    parser.add_argument('--learning_rate', type=str, default=0.0001, help="learning rate")
    parser.add_argument('--num_epochs', type=str, default=60, help="num pochs")
    parser.add_argument('--lyc2vec', type=str, default=50, help="num pochs")
    opt = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # save np.load
    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

    dataset = np.load(opt.data)

    syll_model_path = 'Skip-gram_lyric_encoders/syllEncoding_skipgram_dim_50.bin'
    word_model_path = 'Skip-gram_lyric_encoders/wordLevelEncoder_skipgram_dim_50.bin'
    syllModel = Word2Vec.load(syll_model_path)
    wordModel = Word2Vec.load(word_model_path)
    
    lyric_vocabulary = syllModel.wv.key_to_index
    #music_vocabulary = build_vocabulary(dataset) #len=6064
    music_vocabulary_file = 'saved_model/music_vocabulary_'+str(opt.seqlen)+'.npy'
    music_vocabulary = np.load(music_vocabulary_file)
    music_vocabulary = music_vocabulary.item()

    dtrain_set = TxtDatasetProcessing(dataset, syllModel, wordModel, opt.seqlen, opt.lyc2vec, music_vocabulary)
    #data = dtrain_set[0]
    train_loader = DataLoader(dtrain_set, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=4)

    encoder = Encoder(input_size=opt.lyc2vec*2, hidden_size=256, num_layers=4, vocabulary=lyric_vocabulary).to(device)
    decoder = Decoder(embedding_dim=100, hidden_size=256, num_layers=4, vocabulary=music_vocabulary).to(device)

    model = Seq2Seq(encoder, decoder).to(device)
    model.apply(init_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(encoder.parameters(), lr=opt.learning_rate)

    for epoch in range(opt.num_epochs):
        model.train()
        en_state_h = encoder.init_state(opt.batch_size)
        en_state_h = Variable(en_state_h.to(device))
        optimizer = adjust_learning_rate(optimizer, epoch, opt.learning_rate)
        for iter, traindata in enumerate(train_loader):
            lyric_input, lyric_label, music_input, music_label = traindata
            lyric_input = Variable(lyric_input.transpose(0, 1).to(device))
            lyric_label = Variable(lyric_label.transpose(0, 1).to(device))
            music_input = Variable(music_input.transpose(0, 1).to(device))
            music_label = Variable(music_label.transpose(0, 1).to(device))

            optimizer.zero_grad()

            en_pred, de_pred, en_state_h = model(lyric_input, music_input, en_state_h)
            en_dim = en_pred.shape[-1]
            de_dim = de_pred.shape[-1]
            en_pred = en_pred.view(-1, en_dim)
            de_pred = de_pred.view(-1, de_dim)
            lyric_label = lyric_label.reshape(-1)
            music_label = music_label.reshape(-1)
            en_loss = criterion(en_pred, lyric_label)
            de_loss = criterion(de_pred, music_label)
            loss = en_loss + de_loss

            loss.backward()
            optimizer.step()

            en_state_h = en_state_h.detach()
            if iter % 200 == 0:
                print({ 'epoch': epoch, 'batch': iter, 'loss': loss.item()})


        with torch.no_grad():
            print('epoch:', epoch)
            model.eval()
            lyrics = [['I'], ['give'], ['you'], ['my'], ['hope'], ['I`ll'], ['give'], ['you'], ['my'], ['dreams'], ['I`m'], ['gi', 'ving'], ['you'], ['a', 'ny', 'thing'], ['you'], ['need'], ['I`ll'], ['give'], ['you'], ['my']]
            melody = [[[59.0, 0.5, 0.0]], [[59.0, 1.0, 0.0]], [[59.0, 1.0, 0.0]], [[59.0, 1.0, 0.0]], [[68.0, 1.0, 0.0]], [[59.0, 1.0, 2.0]], [[59.0, 0.5, 0.0]], [[59.0, 1.0, 0.0]], [[59.0, 0.5, 0.0]], [[66.0, 1.0, 0.0]], [[59.0, 1.0, 2.0]], [[59.0, 0.5, 0.0], [59.0, 1.0, 0.0]], [[59.0, 1.5, 0.0]], [[71.0, 2.0, 0.0], [69.0, 2.0, 0.0], [68.0, 2.0, 0.0]], [[66.0, 1.0, 0.0]], [[66.0, 2.0, 0.0]], [[64.0, 2.0, 0.0]], [[59.0, 1.0, 0.0]], [[59.0, 0.5, 0.0]], [[59.0, 0.5, 0.0]]]

            lyric_input, lyric_label, music_input, music_label = data_process(lyrics, melody, opt.seqlen, opt.lyc2vec, music_vocabulary)
            lyric_input, lyric_label, music_input, music_label = lyric_input.unsqueeze(0), lyric_label.unsqueeze(0), music_input.unsqueeze(0), music_label.unsqueeze(0)
            lyric_input = Variable(lyric_input.transpose(0, 1).to(device))
            lyric_label = Variable(lyric_label.transpose(0, 1).to(device))
            music_input = Variable(music_input.transpose(0, 1).to(device))
            music_label = Variable(music_label.transpose(0, 1).to(device))
            en_state_h = encoder.init_state(1)
            en_state_h = Variable(en_state_h.to(device))

            en_pred, de_pred, en_state_h = model(lyric_input, music_input, en_state_h)
            en_pred, de_pred = en_pred.squeeze(1), de_pred.squeeze(1)
            predicted_lyrics = torch.argmax(en_pred, dim=1)
            predicted_melody = torch.argmax(de_pred, dim=1)
            sentence = []
            for idx in predicted_lyrics:
                sentence.append(syllModel.wv.index_to_key[int(idx.item())])
            print(sentence, predicted_melody)

        #filename = 'generator_'+'seqlen_'+str(opt.seqlen)+'_embed_'+str(opt.lyc2vec)+'.pkl'
        filename = 'generator_'+'seqlen_'+str(opt.seqlen)+'_embed_'+str(opt.lyc2vec)+'epoch'+str(epoch)+'.pkl'
        torch.save(model.state_dict(), filename)
        print('File %s is saved.' % filename)