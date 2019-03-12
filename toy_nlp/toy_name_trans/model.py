import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim


class Translator(nn.Module):

    def __init__(self, embedding_dim=10):
        super(Translator, self).__init__()

        # 所有的字符被拆散成字符向量
        # 英文a-z，全部转小写

        self.char_embedding = (26, embedding_dim)
        self.nn_lstm = ()