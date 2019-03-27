import os, random
from data import *

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

MAX_LENGTH = 10


class TranslatorEncoder(nn.Module):

    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(TranslatorEncoder, self).__init__()

        # 所有的字符被拆散成字符向量
        # 英文a-z，全部转小写
        
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_sequence, input_lengths, hidden=None):
        embedded = self.embedding(input_sequence)
        # 打包padding
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        # 解包
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)

        # bidirectional 求和
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.method = method
        if self.method not in ["dot", "general", "concat"]:
            raise ValueError(self.method, " not is a valid attention method.")
        self.hidden_size = hidden_size
        if self.method == "general":
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == "concat":
            self.attn = nn.Linear(self.hidden_size*2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden*encoder_output, dim=2)
    
    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden*energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(hidden*energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        attn_energies = getattr(self, self.method+"_score")(hidden, encoder_outputs)
        attn_energies = attn_energies.t()
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class TranslatorDecoder(nn.Module):
    def __init__(self, attn_method, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(TranslatorDecoder, self).__init__()

        self.attn_method = attn_method
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size*2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
        self.attn = Attention(attn_method, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)

        rnn_output, hidden = self.gru(embedded, last_hidden)
        attn_weights = self.attn(rnn_output, encoder_outputs)
        
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        return output, hidden
        
# mask is describing the padding of the target tensor
def maskNLLLoss(input, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(input, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss#.to(device)
    return loss, nTotal.item()

# GreedySearchDecoder is used to evaluation phrase
class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long)

        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)

        for _ in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # torch.max? as next decoder input
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)

            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)

            decoder_input = torch.unsqueeze(decoder_input, 0)

        return all_tokens, all_scores
