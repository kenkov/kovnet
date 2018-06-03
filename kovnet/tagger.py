#! /usr/bin/env python
# coding:utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from kovnet.vectorizer import IDVectorizer
from sklearn.preprocessing import LabelEncoder


class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, label_size):
        super(self.__class__, self).__init__()

        num_layers = 1

        # padding_idx=0 指定で、 入力 0 に対して 0 ベクトルが割り当てられる
        self.emb = nn.Embedding(vocab_size, embedding_dim,
                                padding_idx=0)
        self.rnn = nn.RNN(embedding_dim,  # input dim
                          embedding_dim,  # hidden dim
                          num_layers,
                          dropout=0)
        self.hidden_to_label = nn.Linear(embedding_dim, label_size)

        self.vocab_size = vocab_size
        self.label_size = label_size
        self.embedding_dim = embedding_dim

    def forward(self, vec):
        embedding = self.emb(vec)

        batch_size = vec.shape[1]
        # state shape (num_layers, batch_size, hidden_size)
        hidden_state = torch.zeros(1, batch_size, self.embedding_dim)
        out, hidden_state = self.rnn(embedding, hidden_state)

        label_out = self.hidden_to_label(out)

        return label_out


if __name__ == "__main__":
    texts = ["今日 は 疲れ た ので 帰っ て 来 た",
             "今日 は 雨 だ",
             "明日 は どうなる だろう"]
    num_samples = len(texts)

    # fit vectorizer
    vectorizer = IDVectorizer()
    vectorizer.fit(texts)

    # params
    batch_size = 2
    vocab_size = len(vectorizer.vocabulary)
    embedding_dim = 100
    label_size = vocab_size

    # model
    model = RNNLanguageModel(vocab_size, embedding_dim, label_size)
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(100):
        batch_perm = torch.randperm(num_samples)
        for batch_idx in range(0, num_samples, batch_size):
            # initialize gradient
            model.zero_grad()

            # forward
            X_raw = [texts[i] for i in
                     batch_perm[batch_idx:batch_idx+batch_size]]
            max_len = 10
            X_ = vectorizer.transform(X_raw, max_len)
            X = X_[:-1]
            y = X_[1:]
            print("X.shape {}, y.shape {}".format(X.shape, y.shape))
            y_pred = model(X)
            print("y.shape {}, y_pred.shape {}".format(y.shape, y_pred.shape))

            # calculate loss
            loss = loss_function(y_pred.view(-1, vocab_size),
                                 y.contiguous().view(-1))

            # backward
            loss.backward()

            # update parameter
            optimizer.step()

            # log
            print("Epoch: {}, loss: {}".format(epoch, loss))
