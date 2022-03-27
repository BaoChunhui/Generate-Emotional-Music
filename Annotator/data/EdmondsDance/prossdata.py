# -*- coding: utf-8 -*- 
import csv 
import pdb
import random

#{'0': 'ambiguous', '1': 'negative', '2': 'neutral', '3': 'positive'}

csvfile = open("EdmondsDance.csv", "r")
reader = csv.reader(csvfile)

dataset = []

for item in reader:
    if reader.line_num == 1:
        continue
    lyric = item[3]
    joy = item[4]
    trust = item[5]
    fear = item[6]
    surprise = item[7]
    sadness = item[8]
    disgust = item[9]
    anger = item[10]
    anticipation = item[11]
    label = []
    if surprise == '1':
        label.append('0')
    if fear== '1' or sadness == '1' or disgust == '1' or anger == '1':
        label.append('1')
    if anticipation == '1' or joy == '1' or trust == '1':
    	label.append('3')
    mullabel = ",".join(label)
    if mullabel == '1,3' or mullabel == '0,1,3':
        mullabel = '0'
    if mullabel == '':
    	mullabel = '2'
    lyric = lyric.split('<br>')
    data = ''
    for l in lyric:
        l += '. '
        if len(data) + len(l) > 100:
            dataset.append([data[0:len(data)-1], mullabel])
            data = ''
        data += l
#pdb.set_trace()
csvfile.close()

def subset(alist, idxs):
    '''
        用法：根据下标idxs取出列表alist的子集
        alist: list
        idxs: list
    '''
    sub_list = []
    for idx in idxs:
        sub_list.append(alist[idx])

    return sub_list

def split_list(alist, group_num=4, shuffle=True, retain_left=False):
    '''
        用法：将alist切分成group个子列表，每个子列表里面有len(alist)//group个元素
        shuffle: 表示是否要随机切分列表，默认为True
        retain_left: 若将列表alist分成group_num个子列表后还要剩余，是否将剩余的元素单独作为一组
    '''

    index = list(range(len(alist))) # 保留下标

    # 是否打乱列表
    if shuffle: 
        random.shuffle(index) 
    
    elem_num = len(alist) // group_num # 每一个子列表所含有的元素数量
    sub_lists = {}
    
    # 取出每一个子列表所包含的元素，存入字典中
    for idx in range(group_num):
        start, end = idx*elem_num, (idx+1)*elem_num
        sub_lists['set'+str(idx)] = subset(alist, index[start:end])
    # 是否将最后剩余的元素作为单独的一组
    if retain_left and group_num * elem_num != len(index): # 列表元素数量未能整除子列表数，需要将最后那一部分元素单独作为新的列表
        sub_lists['set'+str(idx+1)] = subset(alist, index[end:])
    
    return sub_lists

sub_lists = split_list(dataset, group_num=10, shuffle=True, retain_left=True)

#len(dataset) = 8988
#len(sub_lists['set9'])
train = sub_lists['set0']+sub_lists['set1']+sub_lists['set2']+sub_lists['set3']+sub_lists['set4']+sub_lists['set5']+sub_lists['set6']+sub_lists['set7']
val = sub_lists['set8']
test = sub_lists['set9']
pdb.set_trace()

with open('EdmondsDance.tsv', 'w') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerows(dataset)

with open('train.tsv', 'w') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerows(train)

with open('dev.tsv', 'w') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerows(val)

with open('test.tsv', 'w') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerows(test)