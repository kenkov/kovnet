#! /usr/bin/env python
# coding:utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from kovnet.vectorizer import IDVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim,
                 hidden_dim, label_size):
        super(self.__class__, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim,
                                      padding_idx=0)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, 1,
                          dropout=0)
        self.out_lin = nn.Linear(hidden_dim, label_size)

    def forward(self, vec, lengths):
        emb = self.embedding(vec)

        # RNN forward
        x = nn.utils.rnn.pack_padded_sequence(emb, lengths)
        rnn_out_, rnn_hidden = self.rnn(x)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out_)

        rnn_out_final = rnn_out[[idx.item()-1 for idx in lengths],
                                range(vec.shape[1])]

        out = self.out_lin(rnn_out_final)

        return out


if __name__ == "__main__":
    texts = ["今日 は 疲れ た の もう 帰 ろう",
             "今日 は 雨 らしい ね",
             "明日 は どうなる だろう"]
    labels = ["a", "b", "c"]
    num_samples = len(texts)

    # fit vectorizer
    vectorizer = IDVectorizer()
    vectorizer.fit(texts)

    # fit labels
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    # params
    batch_size = 2
    vocab_size = len(vectorizer.vocabulary)
    embedding_dim = 100
    hidden_dim = embedding_dim
    label_size = len(label_encoder.classes_)
    print(label_encoder.classes_)

    # model
    model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, label_size)
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(100):
        batch_perm = torch.randperm(num_samples)
        for batch_idx in range(0, num_samples, batch_size):
            # initialize grad
            model.zero_grad()

            # prepare samples
            idxes = batch_perm[batch_idx:batch_idx+batch_size]
            samples = [texts[i] for i in idxes]
            lengths = [len(sample.split(" ")) for sample in samples]
            # 長さで降順ソートする
            sorted_idx = np.argsort(lengths)[::-1]
            samples = [samples[i] for i in sorted_idx]
            targets = [labels[i] for i in sorted_idx]
            lengths = [lengths[i] for i in sorted_idx]

            X = vectorizer.transform(samples)
            y = torch.tensor(label_encoder.transform(targets))
            lengths = torch.tensor(lengths)

            # forward
            out = model.forward(X, lengths)
            loss = loss_function(out, y)
            print("Epoch: {}, loss: {}".format(epoch, loss.item()))

            # backward
            loss.backward()
            optimizer.step()
