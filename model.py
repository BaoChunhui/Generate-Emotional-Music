import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb

class LSTMClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes, lyric_vocabulary):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        vocab_size = len(lyric_vocabulary)
        self.word_embeddings = nn.Embedding(vocab_size, input_size)

    def forward(self, X, device):
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