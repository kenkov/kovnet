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
