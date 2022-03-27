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
import pdb

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, vocabulary):
        super(Encoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocabulary = vocabulary

        self.n_vocab = len(vocabulary)

        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size,num_layers=self.num_layers,dropout=0.2,)
        self.fc = nn.Linear(self.hidden_size, self.n_vocab)

    def forward(self, x, prev_state):
    	#x: seqlen, batch_size, embed_dim
        states, hidden = self.gru(x, prev_state) #states: seqlen, batch_size, hidden_size 
        #hidden: num_layers, batch_size, hidden_size

        logits = self.fc(states) #seqlen, batch_size, vocab_dim

        return logits, hidden, states

    def init_state(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size))
                #torch.zeros(self.num_layers, sequence_length, self.hidden_size))

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]

class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layers, vocabulary):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocabulary = vocabulary
        self.n_vocab = len(vocabulary)
        
        self.embedding = nn.Embedding(num_embeddings=self.n_vocab,embedding_dim=self.embedding_dim)
        self.attention = Attention(self.hidden_size)
        
        self.gru = nn.GRU(input_size=self.embedding_dim+self.hidden_size, hidden_size=self.hidden_size,num_layers=self.num_layers,dropout=0.2)
        self.fc = nn.Linear(self.hidden_size*2, self.n_vocab)
    def forward(self, input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        embedded = self.embedding(input).unsqueeze(0)  # (1,B,N)
        #embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.fc(torch.cat([output, context], 1))
        return output, hidden, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        assert encoder.hidden_size == decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.num_layers == decoder.num_layers, \
            "Encoder and decoder must have equal number of layers!"
    def forward(self, lyric_input, music_input, en_state_h):
        #en_state_h, en_state_c = self.encoder.init_state(self.seqlen)
        #en_pred: seqlen, batch size, vocab size
        #en_state_h: num layers, batch size, hidden size
        #states: seqlen, batch size, hidden size
        en_pred, en_state_h, en_states = self.encoder(lyric_input, en_state_h)
        hidden = en_state_h

        de_pred = Variable(torch.zeros(music_input.size(0), music_input.size(1), self.decoder.n_vocab)).cuda()
        for t in range(music_input.size(0)):
            inputw = Variable(music_input[t, :])
            output, hidden, attn_weights = self.decoder(inputw, hidden, en_states)
            de_pred[t] = output
        #de_pred = self.decoder(music_input, en_state_h)
        return en_pred, de_pred, en_state_h

def adjust_learning_rate(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

class TxtDatasetProcessing(Dataset):
    def __init__(self, dataset, syllModel, wordModel, seqlen, lyc2vec, music_vocabulary):
        self.dataset = dataset
        
        lyrics = list(dataset[1])
        musics = list(dataset[0])
        labels = list(dataset[2])

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
    parser.add_argument('--data', type=str, default='lyrics_datasets_v3/dataset_50_v3.npy', help="Dnd data.")
    parser.add_argument('--batch_size', type=str, default=32, help="batch size")
    parser.add_argument('--seqlen', type=str, default=50, help="seqlen")
    parser.add_argument('--learning_rate', type=str, default=0.0001, help="learning rate")
    parser.add_argument('--num_epochs', type=str, default=60, help="num pochs")
    parser.add_argument('--lyc2vec', type=str, default=128, help="num pochs")
    opt = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # save np.load
    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

    dataset = np.load(opt.data)

    syll_model_path = 'Skip-gram_lyric_encoders/syllEncoding_skipgram_dim_128.bin'
    word_model_path = 'Skip-gram_lyric_encoders/wordLevelEncoder_skipgram_dim_128.bin'
    syllModel = Word2Vec.load(syll_model_path)
    wordModel = Word2Vec.load(word_model_path)

    lyric_vocabulary = syllModel.wv.key_to_index
    #music_vocabulary = build_vocabulary(dataset) #len=6064
    music_vocabulary_file = 'saved_model/music_vocabulary_'+str(opt.seqlen)+'.npy'
    music_vocabulary = np.load(music_vocabulary_file)
    music_vocabulary = music_vocabulary.item()

    dtrain_set = TxtDatasetProcessing(dataset, syllModel, wordModel, opt.seqlen, opt.lyc2vec, music_vocabulary)
    #data = dtrain_set[0] torch.Size([19, 256]) torch.Size([19]) torch.Size([19]) torch.Size([19])
    train_loader = DataLoader(dtrain_set, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=4)
    #for data in train_loader:
    #    print(data) torch.Size([64, 19, 256]) torch.Size([64, 19]) torch.Size([64, 19]) torch.Size([64, 19])
    encoder = Encoder(input_size=opt.lyc2vec*2, hidden_size=256, num_layers=4, vocabulary=lyric_vocabulary).to(device)
    decoder = Decoder(embedding_dim=100, hidden_size=256, num_layers=4, vocabulary=music_vocabulary).to(device)

    model = Seq2Seq(encoder, decoder).to(device)
    model.apply(init_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    softmax = nn.Softmax(dim=0)

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
            #lyric_input: torch.Size([49, 32, 256]) 
            #music_input:torch.Size([49, 32]) 
            #en_state_h:torch.Size([4, 32, 256])
            en_pred, de_pred, en_state_h = model(lyric_input, music_input, en_state_h)

            #unlikelihood loss
            en_loss = 0
            en_pred = en_pred.transpose(0, 1)
            lyric_label = lyric_label.transpose(0, 1)
            for batch in range(opt.batch_size):
                en_pred_batch = en_pred[batch]
                lyric_label_batch = lyric_label[batch]
                for length in range(opt.seqlen-1):
                    logits = en_pred_batch[length]
                    prob = softmax(logits)
                    with torch.no_grad():
                        label = lyric_label_batch[length]
                        negative_samples = list(set(lyric_label_batch[:length].tolist()))
                    likelihood_loss = -1*torch.log(prob[label])
                    unlikelihood_loss = 0
                    if negative_samples:
                        #for ns in negative_samples:
                        unlikelihood_loss = (-1*torch.log(1-prob[negative_samples])).mean()
                        #unlikelihood_loss /= len(negative_samples)
                    en_loss += (likelihood_loss+unlikelihood_loss)
            en_loss /= (opt.batch_size*(opt.seqlen-1))

            #en_dim = en_pred.shape[-1]
            de_dim = de_pred.shape[-1]
            #en_pred = en_pred.view(-1, en_dim)
            de_pred = de_pred.view(-1, de_dim)
            #lyric_label = lyric_label.reshape(-1)
            music_label = music_label.reshape(-1)
            #en_loss = criterion(en_pred, lyric_label)
            de_loss = criterion(de_pred, music_label)

            loss = en_loss + de_loss

            loss.backward()
            optimizer.step()

            en_state_h = en_state_h.detach()
            if iter % 100 == 0:
                print({ 'epoch': epoch, 'batch': iter, 'loss': loss.item()})

        filename = 'GRU_generator_'+'seqlen_'+str(opt.seqlen)+'_embed_'+str(opt.lyc2vec)+'_epoch_'+str(epoch)+'.pkl'
        torch.save(model.state_dict(), filename)
        print('File %s is saved.' % filename)