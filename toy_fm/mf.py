#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   mf.py
@Time    :   2019/09/21 17:39:20
@Author  :   gajanlee 
@Version :   1.0
@Contact :   lee_jiazh@163.com
@Desc    :   矩阵分解
'''

import numpy as np
from itertools import product

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    """
    matrix factorization


    Reference:
    代码参考
    http://www.quuxlabs.com/wp-content/uploads/2010/09/mf.py_.txt

    公式推导
    https://hpu-yz.github.io/2019/07/23/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E4%B9%8B%E7%9F%A9%E9%98%B5%E5%88%86%E8%A7%A3(MF)%E5%8F%8A%E5%85%B6python%E5%AE%9E%E7%8E%B0/
    """

    n, m = R.shape

    Q = Q.T


    for step in range(steps):
        for (i, j) in product(range(n), range(m)):
            if R[i, j] > 0:
                # e_ij_2 = (r_ij - r_ij_hat)^2
                e_ij = R[i, j] - np.dot(P[i, :], Q[:, j])
                for k in range(K):
                    P[i, k] -= alpha * (-2 * e_ij * Q[k, j] + beta * P[i, k])
                    Q[k, j] -= alpha * (-2 * e_ij * P[i, k] + beta * Q[k, j])

        e = 0
        for (i, j) in product(range(n), range(m)):
            if R[i, j] > 0:
                e_ij = R[i, j] - np.dot(P[i, :], Q[:, j])
                e += pow(R[i, j] - np.dot(P[i, :], Q[:, j]), 2)
                e += sum(beta/2 * (pow(P[i, k], 2) + pow(Q[k, j], 2)) for k in range(K))

        if step % 10 == 0:
            print(f"step {step}: error is {e}")
            
        if e < 0.01:
            break
    return P, Q.T



##### test #####
if __name__ == "__main__":
    R = [
        [5,3,0,1],
        [4,0,0,1],
        [1,1,0,5],
        [1,0,0,4],
        [0,1,5,4],
    ]
    R = np.array(R)

    N, M = R.shape
    K = 2
    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)

    nP, nQ = matrix_factorization(R, P, Q, K)

