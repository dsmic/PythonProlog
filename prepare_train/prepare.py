#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:50:23 2019

@author: detlef
"""
#pylint: disable=R0903, C0301, C0103, C0111

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# restarting in the same console throws an tensorflow error, force a new console
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


import os

from random import shuffle

import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Activation, Embedding
from keras.layers import LSTM, CuDNNLSTM
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

LSTM_use = CuDNNLSTM

# uncomment the following to disable CuDNN support
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#LSTM_use = LSTM
###########################################

vocab = {}
count_chars = 0
def add_translate(cc):
    #pylint: disable=W0603
    global count_chars
    vocab[cc] = count_chars
    count_chars += 1

for c in range(ord('a'), ord('z')+1):
    add_translate(chr(c))
for c in range(ord('0'), ord('9')+1):
    add_translate(chr(c))

add_translate(',')
add_translate('[')
add_translate(']')
add_translate('_')
add_translate(' ')

print("num of different chars", len(vocab))

def check_all_chars_in(x):
    for cc in x:
        if cc not in vocab:
            return False
    return True

print(vocab)

depth_num = 3
expression_database = {}

for i in range(1, depth_num + 1):
    print("depth", i)
    count_lines = 0
    count_new_lines = 0
    f1 = open('tttt'+str(i)+'.tmp')
    f2 = open('tttx'+str(i)+'.tmp')
    for line in f1:
        expression = f2.readline().strip()
        count_lines += 1
        if expression not in expression_database:
            count_new_lines += 1
            expression_database[expression] = line.strip()
    print('files num', i, 'lines', count_lines, 'new_lines', count_new_lines)
print('total lines', len(expression_database))

train_data = []
max_length = 0
max_output = 0
for key in expression_database:
    #print(key, expression_database[key])
    tosplit = expression_database[key][1:]
    tosplit = tosplit[:-1].replace("'", "")
    tosplit = tosplit[:-1].replace("(", "")
    splitted = tosplit.split("), ")
    for ob in splitted:
        elements = ob.split(", ")
        if len(elements) == 3:
            if not check_all_chars_in(elements[0]+' '+elements[1]+elements[2]):
                print("chars missing", elements[0], elements[1], elements[2])
            else:
                output = int(elements[1]) - 1 #logging counts from 1, we need 0
                data = elements[0]+" "+elements[2]
                train_data.append((output, data))
                if len(data) > max_length:
                    max_length = len(data)
                if output > max_output:
                    max_output = output

shuffle(train_data)
len_full_data = len(train_data)
len_div_10 = len_full_data // 10
len_rest = len_full_data - len_div_10

valid_data = train_data[(len_full_data-len_div_10):]
train_data = train_data[:-len_div_10]
print("len of train data", len(train_data))
print("len of valid data", len(valid_data))

print("max len of data", max_length, "max output", max_output)

def str_to_int_list(x, ml):
    x = ('{:>'+str(ml)+'30}').format(x)
    ret = []
    for cc in x:
        ret.append(vocab[cc])
    return ret

class KerasBatchGenerator(object):

    def __init__(self, datain, vocabin):
        self.data = datain
        self.vocab = vocabin
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0

    def generate(self):
        while True:
            if self.current_idx >= len(self.data):
                self.current_idx = 0
            #print(self.current_idx,self.data[self.current_idx][1],str_to_int_list(self.data[self.current_idx][1], max_length))
            tmp_y = np.array([self.data[self.current_idx][0]], dtype=int).reshape((1, 1))
            tmp_x = np.array([str_to_int_list(self.data[self.current_idx][1], max_length)], dtype=int).reshape((1, -1))
            self.current_idx += 1
            #print(tmp_x,tmp_y)
            yield tmp_x, to_categorical(tmp_y, num_classes=max_output + 1)


train_data_generator = KerasBatchGenerator(train_data, vocab)
valid_data_generator = KerasBatchGenerator(valid_data, vocab)

#for i in range(2*len(train_data)):
#    print(next(train_data_generator.generate()))


hidden_size = 200

model = Sequential()
model.add(Embedding(len(vocab), len(vocab), embeddings_initializer='identity', trainable=False))
#model.add(Masking())
model.add(LSTM_use(hidden_size, return_sequences=True))
model.add(LSTM_use(max_output + 1, return_sequences=False))
model.add(Activation('softmax'))

optimizer = Adam(lr=0.00001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])

print(model.summary())
print("starting")
checkpointer = ModelCheckpoint(filepath='checkpoints/model-{epoch:02d}.hdf5', verbose=1)

num_epochs = 200

model.fit_generator(train_data_generator.generate(), len(train_data), num_epochs, validation_data=valid_data_generator.generate(), validation_steps=len(valid_data), callbacks=[checkpointer])
