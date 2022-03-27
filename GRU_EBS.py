# -*- coding: utf-8 -*-
import torch
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import pdb
from collections import Counter
import argparse
from torch.autograd import Variable
from queue import PriorityQueue
import operator
import pretty_midi
from LSTM_cls import LSTMClassifier
from GRU_generator import Encoder, Decoder, Seq2Seq
from random import randint

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

parser = argparse.ArgumentParser()
parser.add_argument('--seqlen', type=int, default=50)
parser.add_argument('--lyc2vec', type=int, default=128)
parser.add_argument('--outlen', type=int, default=30)
parser.add_argument('--b1', type=int, default=3)
parser.add_argument('--b2', type=int, default=3)
parser.add_argument('--b3', type=int, default=5)
parser.add_argument('--emotion', default='positive')
parser.add_argument('--output_num', type=int, default=1)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

syll_model_path = 'Skip-gram_lyric_encoders/syllEncoding_skipgram_dim_'+str(args.lyc2vec)+'.bin'
word_model_path = 'Skip-gram_lyric_encoders/wordLevelEncoder_skipgram_dim_'+str(args.lyc2vec)+'.bin'
syllModel = Word2Vec.load(syll_model_path)
wordModel = Word2Vec.load(word_model_path)

seqlen = args.seqlen
lyc2vec = args.lyc2vec

generator_file = 'GRU_generator_seqlen_50_embed_128_epoch_51.pkl'
#generator_file = 'saved_model/'+'GRU_generator_'+'seqlen_'+str(seqlen)+'_embed_'+str(lyc2vec)+'_epoch_59.pkl'
binary_clf_file = 'saved_model/'+'LSTM_datasetlen_'+str(seqlen)+'_fold_7_clf.pkl'

word_vocabulary_file = 'saved_model/word_vocabulary.npy'
word_vocabulary = np.load(word_vocabulary_file)
word_vocabulary = word_vocabulary.item()
    
syllable_vocabulary_file = 'saved_model/syllable_vocabulary.npy'
syllable_vocabulary = np.load(syllable_vocabulary_file)
syllable_vocabulary = syllable_vocabulary.item()
    
music_vocabulary_file = 'saved_model/music_vocabulary_'+str(seqlen)+'.npy'
music_vocabulary = np.load(music_vocabulary_file)
music_vocabulary = music_vocabulary.item()

music_index2note = [x for x in music_vocabulary.keys()]

Ge_lyric_vocabulary = syllModel.wv.key_to_index

len_syllable_vocabulary = len(syllable_vocabulary)
len_word_vocabulary = len(word_vocabulary)
len_music_vocabulary = len(music_vocabulary)
len_Ge_lyric_vocabulary = len(Ge_lyric_vocabulary)

binary_clf = LSTMClassifier(input_txt_size=128, input_mus_size=10, hidden_size=256, num_layers=6, num_classes=2, syllable_vocabulary=syllable_vocabulary, music_vocabulary=music_vocabulary)
binary_clf.load_state_dict(torch.load(binary_clf_file))

encoder = Encoder(input_size=lyc2vec*2, hidden_size=256, num_layers=4, vocabulary=Ge_lyric_vocabulary)
decoder = Decoder(embedding_dim=100, hidden_size=256, num_layers=4, vocabulary=music_vocabulary)
generator = Seq2Seq(encoder, decoder)
generator.load_state_dict(torch.load(generator_file))

generator = generator.to(device)
binary_clf = binary_clf.to(device)

softmax = torch.nn.Softmax(dim=0)

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
    lyric_input = torch.zeros((len(lyric), lyc2vec*2), dtype = torch.float64)
    txt_len = 0
    for i in range(len(lyric)):
        word = lyric[i][0]
        if word in wordModel.wv.index_to_key:
            word2Vec = wordModel.wv[word]
        else:
            for k in wordModel.wv.index_to_key:
                if word in k:
                    word2Vec = wordModel.wv[k]

        if word in syllModel.wv.index_to_key:
            syll2Vec = syllModel.wv[word]
        else:
            continue        

        syllWordVec = np.concatenate((word2Vec,syll2Vec))
        lyric_input[txt_len] = torch.from_numpy(syllWordVec)
        txt_len += 1
    return lyric_input.type(torch.float32)

def compute_emotion_score(classifier, emotion, EBS_sets, candidate_pairs, b3):
    new_EBS_sets = []
    if not classifier:
        for i in range(len(EBS_sets)):
            lyric = EBS_sets[i][0]
            melody = EBS_sets[i][1]
            pair = candidate_pairs[i][randint(0, len(candidate_pairs[i])-1)]
            lyric.append([pair[0]])
            melody.append(pair[1])
            new_EBS_sets.append((lyric, melody, 0.5))
        return new_EBS_sets
    else:
        classifier.eval()
        if emotion in ['negative', 'positive']:
            label=['negative', 'positive'].index(emotion)
        for i in range(len(EBS_sets)):
            lyric = EBS_sets[i][0]
            melody = EBS_sets[i][1]
            pairs = candidate_pairs[i]
            for pair in pairs:
                new_lyric = lyric + [[pair[0]]]
                txt = []
                for syll in new_lyric:
                    txt.append(syllable_vocabulary[syll[0]])
                txt = torch.Tensor(txt)
                txt = txt.type(torch.int64)
                txt = Variable(torch.unsqueeze(txt, dim=0).to(device))
                new_mus = melody+[pair[1]]
                mus = torch.Tensor(new_mus)
                mus = mus.type(torch.int64)
                mus = Variable(torch.unsqueeze(mus, dim=0).to(device))

                output = classifier(txt, mus)
                score = softmax(output.squeeze(0))[label]
                new_EBS_sets.append((new_lyric, new_mus, score.item()))

        new_EBS_sets = sorted(new_EBS_sets, reverse = True, key = lambda x:x[2])
        return new_EBS_sets[:b3]

def emotional_beam_search(seed_lyric, generator, classifier, outlen, b1, b2, b3):
    seed_len = len(seed_lyric)
    generator.eval()
    if classifier is not None:
        classifier.eval()

    seed_music = [randint(0, 100)]
    for i in range(seed_len-1):
        lyric_input = seed_lyric[:(i+1)]
        lyric_input = Embedding_lyrics(lyric_input, syllModel, wordModel)
        lyric_input = torch.unsqueeze(lyric_input, dim=1)
        lyric_input = lyric_input.to(device)

        music_input = seed_music[:(i+1)]
        music_input = torch.Tensor(music_input).type(torch.int64)
        music_input = torch.unsqueeze(music_input, dim=1).to(device)

        en_state_h = encoder.init_state(1)
        en_state_h = Variable(en_state_h.to(device))

        en_pred, de_pred, en_state_h = generator(lyric_input, music_input, en_state_h)
        en_state_h = en_state_h.detach()

        de_pred = torch.squeeze(de_pred, dim=1)
        predict = de_pred[-1]
        next_music = torch.argmax(predict)
        next_music = int(next_music.item())
        seed_music.append(next_music)

    EBS_sets = [(seed_lyric, seed_music, 0.5)]
    while True:
        candidate_pairs = []
        for m in EBS_sets:
            lyric_input = m[0]
            lyric_input = Embedding_lyrics(lyric_input, syllModel, wordModel)
            lyric_input = torch.unsqueeze(lyric_input, dim=1)
            lyric_input = lyric_input.to(device)

            music_input = m[1]
            music_input = torch.Tensor(music_input).type(torch.int64)
            music_input = torch.unsqueeze(music_input, dim=1).to(device)

            en_pred, de_pred, en_state_h = generator(lyric_input, music_input, en_state_h)
            en_state_h = en_state_h.detach()

            en_pred = torch.squeeze(en_pred, dim=1)
            en_pred = en_pred[-1]
            en_pred = softmax(en_pred)

            de_pred = torch.squeeze(de_pred, dim=1)
            de_pred = de_pred[-1]
            de_pred = softmax(de_pred)

            seen_music = set(m[1])
            seen_syll = set()

            for m0 in m[0]:
                seen_syll.add(Ge_lyric_vocabulary[m0[0]])
            en_pred[list(seen_syll)] *= 0.5
            de_pred[list(seen_music)] *= 0.9
            #for m1 in m[1]:
                #de_pred[m1] *= 0.8
            
            

            log_prob_lyric, indexes_lyric = torch.topk(en_pred,b1,dim=0)
            log_prob_music, indexes_music = torch.topk(de_pred,b2,dim=0)
            
            m_pairs = []
            for i in range(b1):
                for j in range(b2):
                    syl = syllModel.wv.index_to_key[indexes_lyric[i].item()]
                    mel = int(indexes_music[j].item())
                    m_pairs.append((syl, mel))
            candidate_pairs.append(m_pairs)

        EBS_sets = compute_emotion_score(classifier, args.emotion, EBS_sets, candidate_pairs, b3)

        if EBS_sets[-1][2] > 0.9 and len(EBS_sets[0][0]) > 22:
            break
        elif len(EBS_sets[0][0]) > outlen:
            break

    for i in range(args.output_num):
        lyric_i = EBS_sets[i][0]
        score_i = EBS_sets[i][2]
        music_i = []
        for idx in EBS_sets[i][1]:
            music_note = music_index2note[idx]
            music_note = music_note.split('^')
            pitch = float(music_note[0][2:])
            duration = float(music_note[1][2:])
            rest = float(music_note[2][2:])
            music_i.append(np.array([pitch, duration, rest]))
        print(lyric_i, music_i, score_i)
        midi_pattern = create_midi_pattern_from_discretized_data(music_i)
        destination = args.emotion+str(i)+'.mid'
        midi_pattern.write(destination)
    return



if args.emotion in ['positive', 'negative']:
    classifier = binary_clf
else:
    classifier = None

seed_lyric0 = [['I'], ['give'], ['you'], ['my']]
seed_lyric1 = [['but'], ['when'], ['you'], ['told'], ['me']]
seed_lyric2 = [['if'], ['I'], ['was'], ['your'], ['man']]
seed_lyric3 = [['I'], ['have'], ['a'], ['dream']]
seed_lyric4 = [['when'], ['I'], ['got'], ['the']]

seed_lyric = [['I'], ['give'], ['you'], ['my']]
emotional_beam_search(seed_lyric, generator, classifier, outlen = args.outlen, b1=args.b1, b2=args.b2, b3=args.b3)