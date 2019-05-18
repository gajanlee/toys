#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   lstm_lm.py
@Time    :   2019/05/06 17:22:58
@Author  :   gajanlee 
@Version :   1.0
@Contact :   lee_jiazh@163.com
@Desc    :   None
'''

import keras
from keras.utils import Sequence
from keras.layers import Dense, LSTM, Input, Embedding, Lambda, TimeDistributed
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
import numpy as np
from keras.utils import np_utils
from functools import partial

class LineGenerator(Sequence):
    def __init__(self, lines):
        self.lines = lines

    def __len__(self):
        """The batch size per batch."""
        return 

    def __getitem__(self, idx):
        pass

BASE_DIR   = "./"
TRAIN_FILE = BASE_DIR + "PTB dataset/ptb.train.txt"
VALID_FILE = BASE_DIR + "PTB dataset/ptb.valid.txt"
TEST_FILE  = BASE_DIR + "PTB dataset/ptb.test.txt"

def build_word_dictionary():
    words_set = set()
    for path in [TRAIN_FILE, VALID_FILE]:
        with open(path) as file:
            for i, line in enumerate(file):
                if not line.strip(): continue
                words = line.strip().split(' ')
                words_set.update(words)

    word_to_id, id_to_word = {'PADDING': 0, '<s>': 1, '</s>': 2}, ['PADDING', '<s>', '</s>']+[None]*len(words_set)
    for id, word in enumerate(sorted(words_set), 1):
        word_to_id[word] = id
        id_to_word[id] = word

    return word_to_id, id_to_word

word_to_id, id_to_word = build_word_dictionary()

def convert_corpus(path):
    X, Y = [], []
    with open(path, encoding="utf-8") as corpusFile:
        for line in corpusFile:
            if not line.strip(): continue
            label, text = line.strip().split('\t')
            text_ids, label_id = convert_text(text), convert_label(label)
            X.append(text_ids)
            Y.append(label_id)

    return X, Y

VOCAB_SIZE = len(word_to_id)
EMBED_DIM = 45
MAX_INPUT_LEN = 30
hidden_size = 50# len(word_to_id) // 4)

BATCH_SIZE = 32
def convert_corpus_trainable(file_path):
    while True:
        with open(file_path) as file:
            Xs, Ys = [], []
            
            for num, line in enumerate(file, 1):
                if not line.strip(): continue
                words = line.strip().split(" ")
                if len(words) + 3 > MAX_INPUT_LEN: continue
                
                words = padding(words)
                words_ids = list(map(word_to_id.get, words))
                #words_cates = list(map(partial(np_utils.to_categorical, num_classes=VOCAB_SIZE), words_ids))
                words_cates = words_ids

                X, Y = words_cates[:MAX_INPUT_LEN], words_cates[1:MAX_INPUT_LEN+1]

                Y = list(map(partial(np_utils.to_categorical, num_classes=VOCAB_SIZE), Y))
                Xs.append(X)
                Ys.append(Y)
                if len(Xs) == 32:
                    yield np.array(Xs), np.array(Ys).reshape(32, MAX_INPUT_LEN, -1)
                    Xs, Ys = [], []

def padding(words):
    words = (['<s>']
            + words
            + ['PADDING']*(MAX_INPUT_LEN-1-len(words))
            + ['</s>'])
    return words

def main(mode="train"):
    embedding = Embedding(VOCAB_SIZE, 
                        EMBED_DIM,
                        input_length=MAX_INPUT_LEN)

    input = Input(shape=(MAX_INPUT_LEN, ), dtype="int32")
    input_emb = embedding(input)
    # 如果是双向LSTM则需要concat前向和后向的向量
    # 如果是多层的话，需要stack到一个list中
    # 但是多层双向会发生语言泄露的情况
    # 使用RepeatVector和return_sequences的区别
    # https://stackoverflow.com/questions/51749404/how-to-connect-lstm-layers-in-keras-repeatvector-or-return-sequence-true
    input_lstm = LSTM(hidden_size, return_sequences=True, kernel_initializer='random_uniform')(input_emb)
    # TimeDistributed的作用
    # https://blog.csdn.net/u012193416/article/details/79477220
    output = TimeDistributed(Dense(VOCAB_SIZE, activation="softmax", kernel_initializer='random_uniform'))(input_lstm)

    model = keras.Model([input], output)
    model.compile(keras.optimizers.Adam(0.001),
                'categorical_crossentropy', ['acc', ])
    model.summary()
    
    if mode == "train":
        #trainX, trainY = convert_corpus_trainable(TRAIN_FILE)()
        #print(trainX.shape)
        #print(trainY.shape)
        model.fit_generator(convert_corpus_trainable(TRAIN_FILE),steps_per_epoch=500, epochs=10)
        model.save("language_model.h5")
    elif model == "test":
        model = load_model("language_model.h5")
        #model.predict()

if __name__ == "__main__":
    main("train")