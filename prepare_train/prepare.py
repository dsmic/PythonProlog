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

from random import shuffle
from random import random, randint

import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential, Model, Input
from keras.layers import Activation, Embedding, Dense, Flatten, GlobalMaxPooling1D, GlobalAveragePooling1D, Lambda
from keras.layers import LSTM, CuDNNLSTM, CuDNNGRU, SimpleRNN, GRU
from keras.optimizers import Adam, SGD, RMSprop, Nadam
from keras.callbacks import ModelCheckpoint
import keras.backend

# uncomment the following to disable CuDNN support
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#LSTM_use = LSTM
###########################################


import argparse
from random import shuffle

parser = argparse.ArgumentParser(description='train recurrent net.')
parser.add_argument('--lr', dest='lr',  type=float, default=1e-3)
parser.add_argument('--epochs', dest='epochs',  type=int, default=50)
parser.add_argument('--hidden_size', dest='hidden_size',  type=int, default=50)
parser.add_argument('--final_name', dest='final_name',  type=str, default='final_model')
parser.add_argument('--pretrained_name', dest='pretrained_name',  type=str, default=None)
parser.add_argument('--attention', dest='attention', action='store_true')
parser.add_argument('--depth', dest='depth',  type=int, default=3)
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--only_one', dest='only_one', action='store_true')
parser.add_argument('--revert', dest='revert', action='store_true')
parser.add_argument('--add_history', dest='add_history', action='store_true')
parser.add_argument('--RNN_type', dest='RNN_type',  type=str, default='CuDNNLSTM')
parser.add_argument('--gpu_mem', dest='gpu_mem',  type=float, default=1)
parser.add_argument('--fill_vars_with_atoms', dest='fill_vars_with_atoms', action='store_true')
parser.add_argument('--rand_atoms', dest='rand_atoms', action='store_true')
parser.add_argument('--float_type', dest='float_type',  type=str, default='float32')

args = parser.parse_args()

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = args.gpu_mem
set_session(tf.Session(config=config))

keras.backend.set_floatx(args.float_type)



RNN_type = {}
RNN_type['CuDNNLSTM'] = CuDNNLSTM
RNN_type['CuDNNGRU'] = CuDNNGRU
RNN_type['GRU'] = GRU
RNN_type['SimpleRNN'] = SimpleRNN

LSTM_use = RNN_type[args.RNN_type]

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

output_stats = {}
depth_num = args.depth
expression_database = {}
expression_num = {}
expression_depth = {} #do not overwrite lower depth

used_atoms = ['xproof', 'new', 'gproof', 'p', 'empty', 'ee', 'nn']

def random_atoms(max_length, number):
  atoms =[]
  for _ in range(number):
    length = randint(1,max_length)
    atom = ''
    for _ in range(length):
      atom+=chr(randint(ord('a'),ord('z')))
    atoms.append(atom)
  return atoms

#handle manual written clause input to clause selection log file

def split_brackets(sss):
    counter = 0
    pos = -1
    pp = 1
    num_found = 0
    for cc in sss:
        if cc == '[':
            counter += 1
        elif cc == ']':
            counter -= 1
        elif cc == ',' and counter == 0:
            num_found += 1
            if num_found == 1:
              pos = pp
              break
        pp += 1
    if pos == -1:
        return [sss,'']
    return [sss[:pos-1],sss[pos:]]

train_data = []
max_length = 0
max_output = 0

only_one_data ={}

for i in range(1, depth_num+1):
  print("depth", i)
  count_lines = 0
  count_new_lines = 0
  f1 = open('tttx'+str(i)+'.tmp')
  for line in f1:
    line = line.strip()
    if args.debug:
       print("\nl#l",line)
    line = line[1:-1]
    while len(line)>0:
      [el,line] = split_brackets(line)
      p= el.find(',')
      output = int(el[1:p])
      if args.add_history:
        data = el[p+1:] + line   #history of the search is kept, you probably have to revert the data additionally
      else:
        data = el[p+1:]
      if args.fill_vars_with_atoms:
        if args.rand_atoms:
          used_atoms = random_atoms(6,7)
        shuffle(used_atoms)
        for ir in range(len(used_atoms)):
          data = data.replace('_'+str(ir),used_atoms[ir])
      if args.debug:
          print(el,"-",output,"-",data)
      if data not in only_one_data:
        if args.only_one:
          only_one_data[data]=1
        train_data.append((int(output), data))
        if len(data) > max_length:
            max_length = len(data)
        if output > max_output:
            max_output = output
        if output in output_stats:
            output_stats[output] += 1
        else:
            output_stats[output]=1
     
  
    



# This was the code for track evaluation
#for i in range(1, depth_num + 1):
#    print("depth", i)
#    count_lines = 0
#    count_new_lines = 0
#    f1 = open('tttt'+str(i)+'.tmp')
#    f2 = open('tttx'+str(i)+'.tmp')
#    for line in f1:
#        if i < 5 or random()>0.95:
#          expression = f2.readline().strip()
#          count_lines += 1
#          if expression not in expression_database:
#              count_new_lines += 1
#              expression_database[expression] = line.strip()
#              expression_num[expression] = 1
#              expression_depth[expression] = i
#          else:
#              #fill with random
#              if i == expression_depth[expression] and expression != 'nn' and expression != 'ee':
#                  expression_num[expression] += 1
#                  #print(expression,expression_num[expression],expression,line.strip())
#                  if random() < 1.0 / float(expression_num[expression]):
#                      expression_database[expression] = line.strip()
#    print('files num', i, 'lines', count_lines, 'new_lines', count_new_lines)
#print('total lines', len(expression_database))



#def remove_first(sss, nums_to_remove):
#    counter = 0
#    pos = 1
#    pp = 1
#    num_found = 0
#    for cc in sss[1:]:
#        if cc == '[':
#            counter += 1
#        elif cc == ']':
#            counter -= 1
#        elif cc == ',' and counter == 0:
#            num_found += 1
#            if num_found == nums_to_remove:
#              pos = pp
#              break
#        pp += 1
#    rest = sss[pos+1:-1]
#    counter = 0
#    pos = 1
#    pp = 1
#    for cc in rest:
#        if cc == '[':
#            counter += 1
#        elif cc == ']':
#            counter -= 1
#        elif cc == ',' and counter == 0:
#            pos = pp
#            break
#        pp += 1
#    return rest[:pos-1]

#train_data = []
#max_length = 0
#max_output = 0

#only_one_data ={}
#for key in expression_database:
#    #print(key, expression_database[key])
#    tosplit = expression_database[key][1:]
#    tosplit = tosplit[:-1].replace("'", "")
#    tosplit = tosplit[:-1].replace("(", "")
#    splitted = tosplit.split("), ")
#    for ob in splitted:
#        elements = ob.split(", ")
#        #print(elements)
#        if len(elements) == 3:
#            if not check_all_chars_in(elements[0]+' '+elements[1]+elements[2]):
#                print("chars missing", elements[0], elements[1], elements[2])
#            else:
#                output = int(elements[1]) - 1 #logging counts from 1, we need 0
#                #print(output)
#                if elements[0] == 'eqn1':
#                    #print(elements[2])
#                    el2 = remove_first(elements[2],2)
#                    data = elements[0]+ " " + el2
#                    #print(data)
#                    shuffle(used_atoms)
#                    for ir in range(len(used_atoms)):
#                      data = data.replace('_'+str(ir),used_atoms[ir])
#                    if data not in only_one_data: # and output != 8: #only for debugging to allow the rnn get 100% accurency
#                      print(data, output)
#                      if output in output_stats:
#                          output_stats[output] += 1
#                      else:
#                          output_stats[output]=1
#                      train_data.append((output, data))
#                      if len(data) > max_length:
#                          max_length = len(data)
#                      if output > max_output:
#                          max_output = output
#                      only_one_data[data]= output

shuffle(train_data)
len_full_data = len(train_data)
len_div_10 = len_full_data // 10
len_rest = len_full_data - len_div_10

valid_data = train_data[(len_full_data-len_div_10):]
train_data = train_data[:-len_div_10]
print("len of train data", len(train_data))
print("len of valid data", len(valid_data))

#print('debugging, maxoutput increased!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#max_output += 1

print("max len of data", max_length, "max output", max_output)
print(output_stats)
sumoutput=0
for i in output_stats:
  sumoutput += output_stats[i]
for i,v in sorted(output_stats.items()):
  print(i,v/sumoutput)



###################################################################
# Network


def attentions_layer(x):
  from keras import backend as K
  x1 = x[:,:,1:]
  x2 = x[:,:,0:1]
  x2 = K.softmax(x2)
#  x2 = keras.backend.print_tensor(x2, str(x2))
#  x1 = keras.backend.print_tensor(x1, str(x1))
  x=x1*x2
#  x = keras.backend.print_tensor(x, str(x))
  return x

hidden_size = args.hidden_size

if args.pretrained_name is not None:
  from keras.models import load_model
  model = load_model(args.pretrained_name)
  print("loaded model",model.layers[0].input_shape[1])
  ml = model.layers[0].input_shape[1]
  if (ml != max_length):
    print("model length",ml,"different from data length",max_length)
    max_length = ml
else:
#  model = Sequential()
#  model.add(Embedding(len(vocab), len(vocab), embeddings_initializer='identity', trainable=False, input_shape=(max_length,)))
#  model.add(LSTM_use(hidden_size, return_sequences=True))
#  model.add(LSTM_use(max_output + 1, return_sequences=False))
#  model.add(Dense(max_output +1))
#  model.add(Activation('softmax'))
  
  inputs = Input(shape=(None,))
  embeds = Embedding(len(vocab), len(vocab), embeddings_initializer='identity', trainable=True)(inputs)
  lstm1 = LSTM_use(hidden_size, return_sequences=True)(embeds)
  if args.attention:
    lstm1b = Lambda(attentions_layer)(lstm1)
  else:
    lstm1b = lstm1
  lstm4 = LSTM_use(hidden_size, return_sequences=False)(lstm1b)
#  x1 = Dense(hidden_size, activation='relu')(lstm4)
#  x2 = Dense(hidden_size, activation='relu')(x1)
#  x3 = Dense(hidden_size, activation='relu')(x2)
  x = Dense(max_output +1)(lstm4)
  predictions = Activation('softmax')(x)
  model = Model(inputs=inputs, outputs=predictions)




import inspect
with open(__file__) as f:
    a = f.readlines()
startline = inspect.currentframe().f_lineno
print(a[startline+1:startline+2])
optimizer = RMSprop(lr=args.lr, rho=0.9, epsilon=None, decay=0)

print("learning rate",keras.backend.eval(optimizer.lr))
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])

print(model.summary())


def str_to_int_list(x, ml):
    # uncomment for reverse
    if args.revert:
      x = x[::-1]
    # uncomment for all the same length
    #x = ('{:>'+str(ml)+'}').format(x[-ml:])
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


print("starting")
checkpointer = ModelCheckpoint(filepath='checkpoints/model-{epoch:02d}.hdf5', verbose=1)

num_epochs = args.epochs

history = model.fit_generator(train_data_generator.generate(), 100000, num_epochs, validation_data=valid_data_generator.generate(), validation_steps=10000, callbacks=[checkpointer])

model.save(args.final_name+'.hdf5')
print(history.history.keys())
