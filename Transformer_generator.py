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
from torch.nn import Transformer
import pdb

def adjust_learning_rate(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

class myTransformer(Transformer):
    def forward(self, src, tgt, src_mask = None, tgt_mask = None,
                memory_mask = None, src_key_padding_mask = None,
                tgt_key_padding_mask = None, memory_key_padding_mask = None):
        if not self.batch_first and src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0):
            raise RuntimeError("the batch number of src and tgt must be equal")
        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return memory, output #encoder output and decoder output

class Transformer_seq2seq(nn.Module):
    def __init__(self, transformer_model, dim, lyric_vocabulary, music_vocabulary):
        super().__init__()
        self.transformer_model = transformer_model
        self.dim = dim
        self.lyric_vocabulary=lyric_vocabulary
        self.music_vocabulary=music_vocabulary
        self.lyric_vocab = len(lyric_vocabulary)
        self.music_vocab = len(music_vocabulary)
        self.fc_lyric = nn.Linear(self.dim, self.lyric_vocab)
        self.fc_music = nn.Linear(self.dim, self.music_vocab)
        self.music_embedding = nn.Embedding(num_embeddings=self.music_vocab,embedding_dim=self.dim)

    def forward(self, src, tgt):
        tgt = self.music_embedding(tgt) #torch.Size([64, 19, 256])
        en_hi, de_hi = self.transformer_model(src, tgt) #torch.Size([64, 19, 256])
        en_output = self.fc_lyric(en_hi) #torch.Size([64, 19, 20934])
        de_output = self.fc_music(de_hi) 

        return en_output, de_output

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

        word_input = torch.LongTensor(np.zeros(self.seqlen-1, dtype=np.int64))
        syll_input = torch.LongTensor(np.zeros(self.seqlen-1, dtype=np.int64))

        txt_len = 0
        for i in range(len(lyric)):
            word = ''
            for syll in lyric[i]:
                word += syll
            if word in wordModel.wv.index_to_key:
                word2Vec = wordModel.wv[word]
                word2idx = wordModel.wv.key_to_index[word]
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
                    word_input[txt_len] = word2idx
                    syll_input[txt_len] = syll2idx
                    music_input[txt_len] = note2idx
                if txt_len<self.seqlen and txt_len>0:
                    lyric_label[txt_len-1] = syll2idx
                    music_label[txt_len-1] = note2idx
                txt_len += 1

            if txt_len >= self.seqlen:
                break
            if txt_len >= self.seqlen:
                break
        # word_input.type(torch.int64), syll_input.type(torch.int64), 
        return lyric_input.type(torch.float32), lyric_label.type(torch.int64), music_input.type(torch.int64), music_label.type(torch.int64)

    def __len__(self):
        return len(self.lyrics)

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
    for data in train_loader:
        lyric_input, lyric_label, music_input, music_label = data
        break
        #torch.Size([64, 19, 256]) torch.Size([64, 19])

    transformer_model = myTransformer(d_model=opt.lyc2vec*2, nhead=16, num_encoder_layers=12, num_decoder_layers=12, batch_first=True)
    model = Transformer_seq2seq(transformer_model=transformer_model, dim = opt.lyc2vec*2, lyric_vocabulary=lyric_vocabulary, music_vocabulary=music_vocabulary)
    model = model.to(device)
    
    #out = transformer_model(src, tgt) #torch.Size([64, 19, 256]) torch.Size([64, 19, 256])

    #encoder = Encoder(input_size=opt.lyc2vec*2, hidden_size=256, num_layers=4, vocabulary=lyric_vocabulary).to(device)
    #decoder = Decoder(embedding_dim=100, hidden_size=256, num_layers=4, vocabulary=music_vocabulary).to(device)

    #model = Seq2Seq(encoder, decoder).to(device)
    #model.apply(init_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    softmax = nn.Softmax(dim=0)

    for epoch in range(opt.num_epochs):
        model.train()
        optimizer = adjust_learning_rate(optimizer, epoch, opt.learning_rate)
        for iter, traindata in enumerate(train_loader):
            lyric_input, lyric_label, music_input, music_label = traindata
            lyric_input = Variable(lyric_input.to(device))
            lyric_label = Variable(lyric_label.to(device))
            music_input = Variable(music_input.to(device))
            music_label = Variable(music_label.to(device))

            optimizer.zero_grad()

            en_pred, de_pred = model(lyric_input, music_input)

            #unlikelihood loss
            en_loss = 0
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

            if iter % 100 == 0:
                print({ 'epoch': epoch, 'batch': iter, 'loss': loss.item()})
                #filename = 'Transformer_generator_'+'iter_'+str(iter)+'_epoch_'+str(epoch)+'.pkl'
                #torch.save(model.state_dict(), filename)
                #print('File %s is saved.' % filename)

        filename = 'Transformer_generator_'+'seqlen_'+str(opt.seqlen)+'_embed_'+str(opt.lyc2vec)+'_epoch_'+str(epoch)+'.pkl'
        torch.save(model.state_dict(), filename)
        print('File %s is saved.' % filename)