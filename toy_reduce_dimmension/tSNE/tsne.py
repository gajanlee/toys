#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   tSNE.py
@Time    :   2019/10/04 15:00:03
@Author  :   gajanlee 
@Version :   1.0
@Contact :   lee_jiazh@163.com
@Desc    :   t-SNE implementation
'''
import numpy as np

np.random.seed(1)

sign = np.sign


class tSNE:
    """
    SNE更关注局部特征，因为正态分布长尾概率很小。
    tSNE使用t分布，相当于无数个高斯分布的和，长尾概率更大，簇间更紧凑，不相似的点更远。
    """
    def __init__(self):
        self.dims = 2
        self.perplexity = 30.
        self.tolerance = 1e-5
        self.max_iter = 1000
        
        
    def fit_transform(self, X):
        num, dim = X.shape
        Y = np.random.rand((num, self.dims))
        
        P = self._calculate_pair_probs(X)
        P += np.transpose(P)
        P /= np.sum(P)
        # early exaggeration
        P *= 4
        P = np.maximum(P, 1e-12)
        
        eta = 500
        init_momentum, final_momentum = .5, .8
        
        y_grad = np.empty((num, self.dims))
        gains = np.empty((num, 1))
        
        for iter in tqdm(range(self.max_iter)):
            
            # t distribution
            _Q = 1 / (1 + self._calculate_pair_distance(Y))
            # TODO: check the dialog whether it is zero.
            Q = Q / np.sum(_Q)
            Q = np.maximum(Q, 1e-12)
            
            for i in range(n):
                y_grad[i, :] = np.sum((P - Q) * _Q[:, i] * (Y[i, :] - Y), 0)
            
            # Actually, y_grad is -y_grad
            # minumize the KL divergance, so we need to substract(add negative).
            # Different with the paper
            # Therefore, "sign" not equal will increase gains
            gains = ((gains + .2) * (sign(y_grad) != sign(y_inc)) + 
                    (gains * .8) * (sign(y_grad) == sign(y_inc)))
            momentum = init_momentum if iter < 20 else final_momentum
            # add negative gradient of y
            y_inc = momentum * before_y - eta * (gains * y_grad)
            Y += y_inc
            # normalize y to the center point.
            Y = Y - np.tile(np.mean(y, 0), (n, 1))

            if (iter+1) % 100 == 0:
                loss = (np.sum(P * np.log(P / Q)) if iter > 100
                        else np.sum((P/4) * np.log(P/4 / Q)))
                print(f"Iter {iter}: loss is {loss}")
            if iter == 100:
                P = P / 4   # Stop Extraggation
        
        return Y


            
                
            
        
        
    def _calculate_pair_distance(self, _X):
        """计算距离
        _X : matrix, shape is (num, feats)
        return : distance between pairs, shape is (num, num)
        
        formula: (a - b)^2 = a^2 + b^2 - 2ab
        """
        sum_X = np.sum(np.square(_X), axis=1)
        distance_matrix = sum_X + sum_X.T - 2 * _X * _X.T
        return distance_matrix
    
    def _calculate_pair_probs(self, _X):
        """
        使用二分法来寻找每个样本点合适的高斯分布方差
        """
        num, dim = _X.shape
        distance_matrix = self._calculate_pair_distance(_X)
        pair_probs = np.zeros((num, num))
        sigmas = np.ones((num, 1))
        
        least_perplexity = np.log(self.perplexity)
        
        for idx in tqdm(range(num)):
            
            _sigma, _sigma_max, _sigma_min = sigmas[idx], nf.inf, -nf.inf
            is_number = lambda n: not (n == nf.inf or n == -nf.inf)
            for _ in range(50):
                perplexity, probability = self._calculate_perplexity(distance_matrix[idx],
                                                                idx, sigmas[idx])
                perplexity_diff = perplexity - least_perplexity
                if abs(perplexity_diff) < self.tolerance: break
            
                if perplexity_diff < 0:
                    _sigma_min = _sigma
                    _sigma = (_sigma*2) if is_number(_sigma_max) else (_sigma + sigma_max) / 2
                else:
                    _sigma_max = _sigma
                    _sigma = (_sigma/2) if is_number(_sigma_min) else (_sigma + _sigma_min) / 2
                # update sigma, and start with the new value
                sigmas[idx] = _sigma
            
            pair_probs[idx, ] = probability
        
        # sigma = 1 / sigma^2 与原始的对应关系
        print(f"The mean value of sigmas is {np.mean(np.sqrt(1 / sigmas))}")
        return pair_probs
            
        
    
    
    def _calculate_perplexity(self, distance, idx, sigma):
        """给定一个点和寻找到的（方差值平方分之一），根据概率分布（距离）计算困惑度
        perplexity指的是entropy，并没有进行指数操作(简化运算)。
        """
        prob = np.exp(-distance * sigma)
        prob[idx] = 0
        sum_prob = np.sum(prob)
        perplexity = np.log(sum_prob) + sigma * np.sum(distance*prob) / sum_prob
        probability = prob / sum_prob
        return perplexity, probability
        
        
        
        