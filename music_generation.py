# -*- coding: utf-8 -*-
import torch
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from model import Encoder, Decoder, Seq2Seq, LSTMClassifier
import pdb
from collections import Counter
import argparse
from torch.autograd import Variable
from queue import PriorityQueue
import operator
import pretty_midi

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

parser = argparse.ArgumentParser()
parser.add_argument('--seqlen', type=int, default=20)
parser.add_argument('--lyc2vec', type=int, default=128)
parser.add_argument('--outlen', type=int, default=30)
parser.add_argument('--beam_width', type=int, default=3)
parser.add_argument('--emotion', default='negative')
parser.add_argument('--topk', type=int, default=1)
parser.add_argument('--lam', type=int, default=1)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

syll_model_path = 'Skip-gram_lyric_encoders/syllEncoding_skipgram_dim_'+str(args.lyc2vec)+'.bin'
word_model_path = 'Skip-gram_lyric_encoders/wordLevelEncoder_skipgram_dim_'+str(args.lyc2vec)+'.bin'
syllModel = Word2Vec.load(syll_model_path)
wordModel = Word2Vec.load(word_model_path)

seqlen = args.seqlen
lyc2vec = args.lyc2vec

generator_file = 'saved_model/'+'generator_'+'seqlen_'+str(seqlen)+'_embed_'+str(lyc2vec)+'.pkl'
#generator_file = 'generator_seqlen_20_embed_50epoch99.pkl'
binary_clf_file = 'saved_model/binaryEmotion_datasetlen_20_fold_0_clf_LSTM_.pkl'
#ekman_clf_file = 'saved_model/'+'ekman_'+'seqlen_'+str(seqlen)+'_embed_'+str(lyc2vec)+'.pkl'
music_vocabulary_file = 'saved_model/music_vocabulary_'+str(seqlen)+'.npy'

lyric_vocabulary = syllModel.wv.key_to_index
music_vocabulary = np.load(music_vocabulary_file)
music_vocabulary = music_vocabulary.item()

music_index2note = [x for x in music_vocabulary.keys()]

clf_lyric_vocabulary_file = 'saved_model/lyric_vocabulary_clf.npy'
clf_lyric_vocabulary = np.load(clf_lyric_vocabulary_file)
clf_lyric_vocabulary = clf_lyric_vocabulary.item()

len_lyric_vocabulary = len(lyric_vocabulary)
len_music_vocabulary = len(music_vocabulary)

encoder = Encoder(input_size=lyc2vec*2, hidden_size=256, num_layers=4, vocabulary=lyric_vocabulary)
decoder = Decoder(embedding_dim=100, hidden_size=256, num_layers=4, vocabulary=music_vocabulary)
generator = Seq2Seq(encoder, decoder)
#ekman_clf = LSTMClassifier(input_size=lyc2vec+3, hidden_size=256, num_layers=6, num_classes=6)
binary_clf = LSTMClassifier(input_size=128, hidden_size=256, num_layers=6, num_classes=2, lyric_vocabulary=clf_lyric_vocabulary)

generator.load_state_dict(torch.load(generator_file))
#ekman_clf.load_state_dict(torch.load(ekman_clf_file))
binary_clf.load_state_dict(torch.load(binary_clf_file))

generator = generator.to(device)
#ekman_clf = ekman_clf.to(device)
binary_clf = binary_clf.to(device)

softmax = torch.nn.Softmax(dim=0)

def get_clf_input(n_lyric, n_melody):
    txt = torch.LongTensor(np.zeros(len(n_lyric), dtype=np.int64))
    for i in range(len(n_lyric)):
        syll = syllModel.wv.index_to_key[n_lyric[i]]
        if syll in clf_lyric_vocabulary.keys():
            syll2idx = lyric_vocabulary[syll]
            txt[i] = syll2idx
    return txt

def compute_emotion_score(classifier, emotion ,n_lyric, n_melody, new_lyric, new_melody):
    classifier.eval()
    if new_lyric is not None:
        n_lyric.append(int(new_lyric.item()))
    if new_melody is not None:
        n_melody.append(int(new_melody.item()))
    
    if emotion in ['negative', 'positive']:
        label=['negative', 'positive'].index(emotion)
    #elif emotion in ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']:
    #    label=['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'].index(emotion)
    clf_input = get_clf_input(n_lyric, n_melody)
    clf_input = Variable(clf_input.unsqueeze(0).to(device))
    output = classifier(clf_input, device)
    score = softmax(output.squeeze(0))[label]
    #print(softmax(output.squeeze(0)))
    predict = torch.argmax(output.squeeze(0))
    
    return score, predict

def create_midi_pattern_from_discretized_data(discretized_sample):
    new_midi = pretty_midi.PrettyMIDI()
    voice = pretty_midi.Instrument(1)  # It's here to change the used instruments !
    tempo = 120
    ActualTime = 0  # Time since the beginning of the song, in seconds
    for i in range(0,len(discretized_sample)):
        length = discretized_sample[i][1] * 60 / tempo  # Conversion Duration to Time
        if i < len(discretized_sample) - 1:
            gap = discretized_sample[i + 1][2] * 60 / tempo
        else:
            gap = 0  # The Last element doesn't have a gap
        note = pretty_midi.Note(velocity=100, pitch=int(discretized_sample[i][0]), start=ActualTime,
                                end=ActualTime + length)
        voice.notes.append(note)
        ActualTime += length + gap  # Update of the time

    new_midi.instruments.append(voice)

    return new_midi

def Embedding_lyrics(lyric, syllModel, wordModel):
    length = 0
    for i in range(len(lyric)):
        for syll in lyric[i]:
            length += 1
    lyric_input = torch.zeros((length, lyc2vec*2), dtype = torch.float64)
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
            if syll in syllModel.wv.index_to_key:
                syll2Vec = syllModel.wv[syll]
            else:
                continue
            syllWordVec = np.concatenate((word2Vec,syll2Vec))
            lyric_input[txt_len] = torch.from_numpy(syllWordVec)
            txt_len += 1
    return lyric_input.type(torch.float32)

class BeamSearchNode(object):
    def __init__(self, hiddenstate, hidden, en_states,previousNode, wordId, musicID, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.hidden = hidden
        self.en_states = en_states
        self.prevNode = previousNode
        self.wordid = wordId
        self.musicid = musicID
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        #reward = 0
        # Add here a function for shaping a reward
        return self.logp / float(self.leng - 1 + 1e-6)# + alpha * reward  # 注意这里是有惩罚参数的，参考恩达的 beam-search

    def __lt__(self, other):
        return self.leng < other.leng  # 这里展示分数相同的时候怎么处理冲突，具体使用什么指标，根据具体情况讨论

    def __gt__(self, other):
        return self.leng > other.leng

def get_music(n):
    lyrics = []
    melody = []
    lyrics.append(n.wordid)
    melody.append(n.musicid)
    # back trace
    while n.prevNode != None:
        n = n.prevNode
        lyrics.append(n.wordid)
        melody.append(n.musicid)

    lyrics = lyrics[::-1] #调转顺序
    melody = melody[::-1]

    return lyrics, melody

def emotional_beam_search(seed_lyric, generator, classifier, seed_len = 4, outlen = 20, beam_width=4):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''
    generator.eval()
    if classifier is not None:
        classifier.eval()

    en_state_h = encoder.init_state(1)
    en_state_h = Variable(en_state_h.to(device))

    nodes = PriorityQueue()
    first_syllable = syllModel.wv.key_to_index[seed_lyric[0][0]]
    first_music = 'p_69.0^d_1.0^r_0.0'

    lyric_input = Embedding_lyrics(syllModel.wv.index_to_key[first_syllable], syllModel, wordModel) #4, 100
    lyric_input = torch.unsqueeze(lyric_input, dim=1) #1, 1, 100
    lyric_input = lyric_input.to(device)

    _, en_state_h, en_states = generator.encoder(lyric_input, en_state_h)

    first_music_input = music_vocabulary[first_music]
    
    node = BeamSearchNode(en_state_h, en_state_h, en_states, None, first_syllable, first_music_input, 0, 1)
    nodes.put((-node.eval(), node))
    
    for syll in seed_lyric[1:seed_len]:
        score, n = nodes.get()
        syllid = n.wordid
        musicid = n.musicid
        musicid = torch.Tensor([musicid]).type(torch.int64)
        musicid = Variable(musicid.to(device))
        en_state_h = n.h
        hidden = n.hidden
        en_states1 = n.en_states
        lyric_input = Embedding_lyrics([[syllModel.wv.index_to_key[syllid]]], syllModel, wordModel)
        lyric_input = torch.unsqueeze(lyric_input, dim=1)
        lyric_input = lyric_input.to(device)
        _, en_state_h, en_states2 = generator.encoder(lyric_input, en_state_h)
        en_states = torch.cat((en_states1, en_states2), dim=0)

        output, hidden, _ = generator.decoder(musicid, hidden, en_states)
        predicted = torch.argmax(output, dim=1)
        predicted = int(predicted.item())
        syll2idx = syllModel.wv.key_to_index[syll[0]]
        node = BeamSearchNode(en_state_h, hidden, en_states, n, syll2idx, predicted, 0, n.leng + 1)
        nodes.put((-node.eval(), node))
    
    while True:
        score, n = nodes.get()
        n_lyric, n_melody = get_music(n)
        if classifier is not None:
            emotion_score, predict = compute_emotion_score(classifier, args.emotion ,n_lyric, n_melody, None, None)
            if emotion_score > 0.9 and n.leng > 22:
                print(emotion_score, predict)
                break
        if n.leng == outlen:
            n_lyric, n_melody = get_music(n)
            if classifier is not None:
                emotion_score, predict = compute_emotion_score(classifier, args.emotion ,n_lyric, n_melody, None, None)
                print('emotion_score', emotion_score, predict)
            nodes.put((score, n))
            break

        syllid = n.wordid
        en_state_h = n.h
        musicid = n.musicid
        musicid = torch.Tensor([musicid]).type(torch.int64)
        musicid = Variable(musicid.to(device))
        en_state_h = n.h
        hidden = n.hidden
        en_states1 = n.en_states
        lyric_input = Embedding_lyrics([[syllModel.wv.index_to_key[syllid]]], syllModel, wordModel)
        lyric_input = torch.unsqueeze(lyric_input, dim=1)
        lyric_input = lyric_input.to(device)
        logits, en_state_h, en_states2 = generator.encoder(lyric_input, en_state_h)
        logits = logits.squeeze(0)
        logits = logits.squeeze(0)
        logits = softmax(logits)

        en_states = torch.cat((en_states1, en_states2), dim=0)

        lyric_input = Embedding_lyrics([[syllModel.wv.index_to_key[syllid]]], syllModel, wordModel)
        lyric_input = torch.unsqueeze(lyric_input, dim=1)
        lyric_input = lyric_input.to(device)
        output, hidden, _ = generator.decoder(musicid, hidden, en_states)
        output = output.squeeze(0)
        output = softmax(output)

        output[n_melody] *= 0.8
        logits[n_lyric] = 0      #已经出现过的不能再生成

        log_prob_lyric, indexes_lyric = torch.topk(logits,5,dim=0)
        log_prob_melody, indexes_melody = torch.topk(output,2,dim=0)
        combination = torch.zeros((indexes_lyric.shape[0]*indexes_melody.shape[0], 3))
        scores = torch.zeros((indexes_lyric.shape[0]*indexes_melody.shape[0]))
        idx = 0
        for i in range(indexes_lyric.shape[0]):
            for j in range(indexes_melody.shape[0]):
                if classifier is not None:
                    emotion_score, _ = compute_emotion_score(classifier, args.emotion ,n_lyric, n_melody, indexes_lyric[i], indexes_melody[j])
                else:
                    emotion_score = 0
                score = log_prob_lyric[i]+log_prob_melody[j]+emotion_score * args.lam 
                combination[idx][0] = indexes_lyric[i]
                combination[idx][1] = indexes_melody[j]
                combination[idx][2] = score
                scores[idx] = score
                idx += 1
        log_prob_music, indexes_music = torch.topk(scores,beam_width,dim=0)
        #log_prob, indexes = torch.topk(logits.squeeze(0),beam_width,dim=2)

        nextnodes = []
        for new_k in range(beam_width):
            lyricid = int(combination[int(indexes_music[new_k].item())][0].item())
            melodyid = int(combination[int(indexes_music[new_k].item())][1].item())
            score = combination[int(indexes_music[new_k].item())][2]
            #decoded_t = indexes[0][new_k].view(-1)
            #log_p = log_prob[0][new_k].item()
            node = BeamSearchNode(en_state_h, hidden, en_states, n, lyricid, melodyid, n.logp+score, n.leng + 1)
            #node = BeamSearchNode(en_state_h, n, decoded_t, n.logp + log_p, n.leng + 1)
            score = -node.eval()
            nextnodes.append((score, node))
        for i in range(len(nextnodes)):
            score, nn = nextnodes[i]
            nodes.put((score, nn))

    endnodes = [nodes.get() for _ in range(args.topk)]

    utterances_lyrics = []
    utterances_melody = []
    for score, n in sorted(endnodes, key=operator.itemgetter(0)):
        utterance_lyrics = []
        utterance_melody = []
        utterance_lyrics.append(n.wordid)
        utterance_melody.append(n.musicid)
        # back trace
        while n.prevNode != None:
            n = n.prevNode
            utterance_lyrics.append(n.wordid)
            utterance_melody.append(n.musicid)

        utterance_lyrics = utterance_lyrics[::-1] #调转顺序
        utterance_melody = utterance_melody[::-1]
        utterances_lyrics.append(utterance_lyrics)
        utterances_melody.append(utterance_melody)

    output = []
    for j in range(len(utterances_lyrics)):
        generated_lyric = []
        for i in utterances_lyrics[j]:
            generated_lyric.append(syllModel.wv.index_to_key[i])
        output.append(generated_lyric)
    print(output)
    
    for i in range(len(utterances_melody)):
        music = []
        for idx in utterances_melody[i]:
            music_note = music_index2note[idx]
            music_note = music_note.split('^')
            pitch = float(music_note[0][2:])
            duration = float(music_note[1][2:])
            rest = float(music_note[2][2:])
            music.append(np.array([pitch, duration, rest]))
        print(music)
        midi_pattern = create_midi_pattern_from_discretized_data(music)
        destination = args.emotion+'.mid'
        midi_pattern.write(destination)

    return output, music

seed_lyric0 = [['I'], ['give'], ['you'], ['my']]
seed_lyric1 = [['but'], ['when'], ['you'], ['told'], ['me']]
seed_lyric2 = [['if'], ['I'], ['was'], ['your'], ['man']]
seed_lyric3 = [['I'], ['have'], ['a'], ['dream']]
seed_lyric4 = [['when'], ['I'], ['got'], ['the']]

if args.emotion in ['positive', 'negative']:
    classifier = binary_clf
#elif args.emotion in ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']:
#    classifier = ekman_clf
else:
    classifier = None
'''
lyrics = []
melody = []
for seed_lyric in [seed_lyric0, seed_lyric1, seed_lyric2, seed_lyric3, seed_lyric4]:
    output, music = generated_lyric = emotional_beam_search(seed_lyric, generator, classifier, seed_len = 4, outlen = args.outlen, beam_width=args.beam_width)
    #pdb.set_trace()
    lyrics.append(output)
    melody.append(music)
print(lyrics, melody)
'''
seed_lyric = [['I'], ['am'], ['a'], ['boy']]
output, music = generated_lyric = emotional_beam_search(seed_lyric, generator, classifier, seed_len = 4, outlen = args.outlen, beam_width=args.beam_width)