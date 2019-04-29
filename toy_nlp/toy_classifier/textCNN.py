#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   textCNN.py
@Time    :   2019/04/30 01:09:32
@Author  :   gajanlee 
@Version :   1.0
@Contact :   lee_jiazh@163.com
@Desc    :   None
'''

import tensorflow as tf
from keras.layers import Input, LSTM, Dense, Embedding, Conv1D, MaxPooling1D, Concatenate, BatchNormalization, Dropout
from keras.utils import np_utils

MAX_LEN = 70

TRAIN_FILE = "./Dataset/train.txt"
VALID_FILE = "./Dataset/validation.txt"
TEST_FILE  = "./Dataset/test.txt"



def build_word_dictionary():
    words_set = set()
    for path in [TRAIN_FILE, VALID_FILE]:
        with open(path) as file:
            for i, line in enumerate(file):
                if not line.strip(): continue
                _, text = line.strip().split('\t')
                words = text.split(' ')
                words_set.update(words)

    word_to_id, id_to_word = {'PADDING': 0}, ['PADDING']+[None]*len(words_set)
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

def convert_label(label):
    label = int(label)
    return label

def convert_text(text):
    tokens = text.split(' ')
    token_ids = list(map(convert_words_to_id, tokens))
    return padding(token_ids)

def convert_words_to_id(words):
    return [word_to_id.get(word, 0) for word in words]

def padding(token_ids):
    token_ids = token_ids[:MAX_LEN]
    token_ids = token_ids + [0]*(MAX_LEN - len(token_ids))
    return token_ids


class TextCnnModel:
    
    def __init__(self):
        
        #self.embedding = Embedding(name="embed",
        #                        weights=[embed_mat], trainable=True)
        self.embedding = Embedding(1000, 64, input_length=10)

        input = Input(shape=(10,), dtype="int32")
        input = self.embedding(input)
        conv = Conv1D(filters=20, kernel_size=3, strides=1, padding="same",                        activation="relu")(input)
        
        # shape  = (?, length, 20)
        # gmp = MaxPooling1D(pool_size=1, strides=1, padding="same")(conv)
        # global max pooling, shape = (?, 20)
        gmp = tf.reduce_max(conv, reduction_indices=[1], name="gmp")
        gmp = Dropout(0.2)(gmp)

        hidden = Dense(50, name="fc1", activation="relu")(gmp)
        hidden = Dropout(0.2)(hidden)
        output = Dense(2, activation="softmax")(hidden)
        self.model = keras.Model([input], output)
        self.model.compile(keras.optimizers.Adam(0.001),
                            'categorical_crossentropy', ['acc', ])
        self.model.summary()

    def train(self, train_X, train_Y, val_X, val_Y):
        self.model.fit(train_X, train_Y, 
                        batch_size=32, epochs=100, 
                        verbose=1, 
                        validation_data=(val_X, val_Y))
        
        score = self.model.evaluate(val_X, val_Y, verbose=1, batch_size=32)
        print(f"validation score is {score}")
    
    def predict_all(self, test_X, test_Y):
        score = self.model.evaluate(test_X, test_Y, verbose=1, batch_size=32)
        print(f"test set score is {score}")

    def save(self):
        self.model.save("sem_classifier.model")

    def predict(self, text):
        if type(text) is str:
            import jieba as jb
            text = list(jb.cut(text))

trainCorpus = convert_corpus(TRAIN_FILE)
trainX, trainY = trainCorpus
trainY = np_utils.to_categorical(trainY, num_classes=2)

valCorpus = convert_corpus(VALID_FILE)
valX, valY = trainCorpus
valY = np_utils.to_categorical(valY, num_classes=2)

m = TextCnnModel()
m.train(trainX, trainY, valX, valY)

# TODO： 导入词向量