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


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim,
                 hidden_dim, label_size):
        super(self.__class__, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim,
                                      padding_idx=0)
        self.rnn = PadRNN(embedding_dim, hidden_dim, 1,
                          dropout=0, rnn_class=nn.GRU)
        self.out_lin = nn.Linear(hidden_dim, label_size)

    def forward(self, vec, lengths):
        emb = self.embedding(vec)

        # RNN forward
        rnn_out, rnn_hidden, rnn_out_final = self.rnn(emb, lengths)
        out = self.out_lin(rnn_out_final)

        return out


if __name__ == "__main__":
    texts = ["明日 は 晴れる", "今日 は 暑い です", "疲れ た ので 早く に 帰り たい です"]
    labels = [1, 0, 0]

    num_samples = len(texts)

    # fit vectorizer
    vectorizer = IDVectorizer()
    vectorizer.fit(texts)

    # fit labels
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    # params
    batch_size = 20
    vocab_size = len(vectorizer.vocabulary)
    embedding_dim = 64
    hidden_dim = embedding_dim
    label_size = len(label_encoder.classes_)

    print("Samples: {}".format(num_samples))
    print("vocab_size: {}, label_size: {}".format(vocab_size, label_size))
    print(label_encoder.classes_)

    # model
    model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, label_size)
    # ignore_idx は指定しない
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(100):
        batch_perm = torch.randperm(num_samples)
        epoch_loss = 0
        for batch_idx in range(0, num_samples, batch_size):
            # initialize grad
            model.zero_grad()

            # prepare samples
            idxes = batch_perm[batch_idx:batch_idx+batch_size]
            samples = [texts[i] for i in idxes]
            targets = [labels[i] for i in idxes]
            lengths = [len(sample.split(" ")) for sample in samples]
            # 長さで降順ソートする
            lengths, samples, targets = sort_by_length(lengths,
                                                       samples,
                                                       targets)

            # Tensor に変換
            X = vectorizer.transform(samples)
            y = torch.tensor(label_encoder.transform(targets))
            lengths = torch.tensor(lengths)

            # forward
            out = model.forward(X, lengths)
            loss = loss_function(out, y)

            # backward
            loss.backward()
            optimizer.step()

            # log
            epoch_loss += loss.item()
            # print("Epoch: {}, loss: {:.6f}".format(epoch, loss.item()))
            # sample_idx = random.choice(range(X.shape[1]))
            # print("> {}\n= {}\n< {}".format(
            #     samples[sample_idx],
            #     targets[sample_idx],
            #     label_encoder.classes_[out[sample_idx].topk(1)[1].item()],
            #     ))
        print("Epoch: {}, loss: {}".format(epoch, epoch_loss))
