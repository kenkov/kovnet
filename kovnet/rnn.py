#! /usr/bin/env python
# coding:utf-8


import torch.nn as nn


class PadRNN(nn.Module):
    def __init__(self, *args, rnn_class=nn.RNN, **kwargs):
        super(self.__class__, self).__init__()

        self.rnn = rnn_class(*args, **kwargs)

    def forward(self, vec, lengths):
        x = nn.utils.rnn.pack_padded_sequence(vec, lengths)
        rnn_out_, rnn_hidden = self.rnn(x)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out_)

        rnn_out_final = rnn_out[[idx.item()-1 for idx in lengths],
                                range(vec.shape[1])]

        return rnn_out, rnn_hidden, rnn_out_final


class Decoder(nn.Module):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 hidden_size,
                 num_layers=1,
                 dropout=0):
        super(self.__class__, self).__init__()

        # hyperparameters
        self.embedding = nn.Embedding(num_embeddings,
                                      embedding_dim,
                                      padding_idx=0)
        self.rnn = PadRNN(embedding_dim, hidden_size, num_layers,
                          dropout=dropout, rnn_class=nn.GRU)
        self.out_lin = nn.Linear(hidden_size, num_embeddings)

    def forward(self, vec, lengths):
        emb = self.embedding(vec)
        rnn_out, rnn_hidden, rnn_out_final = self.rnn(emb, lengths)
        out = self.out_lin(rnn_out)

        return out