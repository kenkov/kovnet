#! /usr/bin/env python
# coding:utf-8


import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from kovnet.vectorizer import IDVectorizer
from kovnet.utils import sort_by_length
from kovnet.rnn import PadRNN
from sklearn.preprocessing import LabelEncoder


class RNNGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim,
                 hidden_dim,):
        super(self.__class__, self).__init__()

        # hyperparameters
        num_layers = 1
        dropout_ratio = 0

        self.embedding = nn.Embedding(vocab_size, embedding_dim,
                                      padding_idx=0)
        self.rnn = PadRNN(embedding_dim, hidden_dim, num_layers,
                          dropout=dropout_ratio, rnn_class=nn.GRU)
        self.out_lin = nn.Linear(hidden_dim, vocab_size)

    def forward(self, vec, lengths):
        emb = self.embedding(vec)
        rnn_out, rnn_hidden, rnn_out_final = self.rnn(emb, lengths)
        out = self.out_lin(rnn_out)

        return out


if __name__ == "__main__":
    texts = ["今日 は 寒い",
             "今日 は 寒い です",
             "明日 は 暑い",
             ]
    num_samples = len(texts)

    vectorizer = IDVectorizer()
    vectorizer.fit(texts)

    # hyperparameters
    batch_size = 2
    embedding_dim = 128
    hidden_dim = 128
    vocab_size = len(vectorizer.vocabulary)

    # model
    model = RNNGenerator(vocab_size, embedding_dim, hidden_dim)
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(1000):
        shuffled_idx = torch.randperm(num_samples)
        for batch_idx in range(0, num_samples, batch_size):
            # initialize grad
            model.zero_grad()

            # サンプルを抽出
            idxes = shuffled_idx[batch_idx:batch_idx+batch_size]
            samples = [texts[i] for i in idxes]
            lengths = [len(sample.split(" ")) for sample in samples]
            lengths, samples = sort_by_length(lengths, samples)

            # ベクトルに変換
            X_ = vectorizer.transform(samples)
            X = X_[:-1]
            y = X_[1:]
            lengths = torch.tensor(lengths) - 1  # minus 1 to remove EOS
            # forward
            out = model(X, lengths)
            loss = loss_function(out.view(-1, vocab_size),
                                 y.contiguous().view(-1))
            
            # backward
            loss.backward()

            # update parameters
            optimizer.step()

            # log
            print("Epoch: {}, loss: {}".format(epoch, loss))