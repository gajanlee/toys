#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   fm.py
@Time    :   2019/09/21 16:46:03
@Author  :   gajanlee 
@Version :   1.0
@Contact :   lee_jiazh@163.com
@Desc    :   Factorization machines
'''

from functools import partial
from math import exp, log
from random import normalvariate #正态分布

import numpy as np
import random

random.seed(1)
np.random.seed(1)

# 数据下的FM所用的sigmoid
def sigmoid_sparse(x):
    return 1 / (1 + exp(-x))

# 设定阈值sigmoid，来适应稠密数据
def sigmoid(x):
    x = max(min(x, 15.), -15.)
    return sigmoid_sparse(x)

# python3默认继承自object
class FM(object):
    """
    FM算法解决了成对特征交互的问题。
    """
    def __init__(self, hidden=2):
        self.hidden = hidden
        self.alpha = 0.01

    def fit(self, X, Y, iters=150):
        """
        X: numpy.matrix

        SGD algorithm to optimize.

        """
        n, m = X.shape
        k = self.hidden
        
        V = normalvariate(0, 0.2) * np.ones((m, k))
        W = np.zeros((m, 1))
        w_0 = 0.
        
        for iter in range(1, iters+1):
            losses = []
            for i in range(n):
                # X[i] shape is (n, )
                label = Y[i]
                predict = self.predict_single(X[i], w_0, W, V)
                loss = self.loss_single(predict, label)

                # update algorithm
                gradient_head = (sigmoid(label * predict) - 1) * label * self.alpha
                w_0 = w_0 - gradient_head * 1
                for j in range(m):
                    if X[i, j] != 0:
                        W[j] -= gradient_head * X[i, j]

                        x_v_mult = (X[i] * V).A.flatten()
                        for k in range(k):
                            V[i, j] -= gradient_head * (X[i, j] * x_v_mult[k] - pow(X[i, j], 2) * V[j, k])
                losses.append(loss)
            if iter % 10 == 0:
                print(f"{iter}: {np.mean(losses)}")


        self.V = V
        self.W = W
        self.w_0 = w_0
        predicts = [self.predict_sigmoid_single(X[i], self.w_0, self.W, self.V)
                    for i in range(n)]


        def predict_correct(prob_label):
            prob, label = prob_label
            return (prob >= 0.5 and label == 1) or (prob < 0.5 and label == -1)

        accuracy = len(list(filter(predict_correct, zip(predicts, Y)))) / n
        print(f"predicting accuracy is {accuracy}")

    def predict_single(self, X, w_0, W, V):
        interaction = np.sum(np.power(X * V, 2) - np.power(X, 2) * np.power(V, 2)) / 2
        return w_0 + np.sum(X * W) + interaction

    def predict_sigmoid_single(self, *args):
        return sigmoid(self.predict_single(*args))

    def get_embedding(self, X):
        # return embedding shape is (n, embedding_size)
        return X * self.V

    def loss_single(self, predict, label):
        return -log(sigmoid(label * predict))

if __name__ == "__main__":
    X = [
        [0.2, 0.3, 0, 0.4, 0.4],
        [0.25, 0.3, 0, 0.45, 0],
        [0.05, 0.2, 0.7, 0.05, 0],
        [0.05, 0.3, 0.6, 0.05, 0],
        [0, 0.2, 0.3, 0.4, 0.1],
    ]
    X = np.matrix(X)

    FM().fit(X, [1, 1, -1, -1, -1])

