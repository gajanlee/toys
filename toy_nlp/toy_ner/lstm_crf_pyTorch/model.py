# conda install pytorch-cpu torchvision-cpu -c pytorch
# https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#bi-lstm-conditional-random-field-discussion

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

START_TAG, STOP_TAG = "<s>", "</s>"

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, dropout=0)

        # A矩阵
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        
        # 状态转移矩阵
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        
        # 起始和终止不存在状态转移
        self.transitions.data[tag_to_ix[START_TAG], :] = -1e4
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -1e4

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def forward_alg(self, feats):
        
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # 起始拥有全部概率
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        forward_var = init_alphas

        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                # 观测分数，emission_score
                # view函数把矩阵转为这个shape
                # expand broadcast 到这个shape
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                
                next_tag_var = forward_var + trans_score + emit_score

                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            
            forward_var = troch.cat(alphas_t).view(1, -1)
        
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embes = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def score_sentence(self, feats, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]])])

        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i+1], tags[i]] + feat[tags[i+1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def viterbi_decode(self, feats):
        pass

        