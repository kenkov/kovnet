#! /usr/bin/env python
# coding:utf-8


import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from kovnet.vectorizer import IDVectorizer
from kovnet.utils import sort_by_length
from kovnet.rnn import Decoder
from sklearn.preprocessing import LabelEncoder


def idx2words(idxes, vocab):
    id2word = {key: word for word, key in vocab.items()}
    return [id2word[idx.item()] for idx in idxes]


def proba2words(proba, vocab):
    idxes = torch.topk(proba, 1, dim=1)[1]
    return idx2words(idxes, vocab)


def load_data(filename):
    with open(filename) as f:
        lst = []
        for line in f:
            text = line.strip("\n")
            words = text.split(" ") + ["</s>"]
            lst.append(" ".join(words))
    return lst


if __name__ == "__main__":
    texts = ["今日 は 寒い </s>",
             "今日 は 寒い です </s>",
             "明日 は 暑い </s>",
             ]
    num_samples = len(texts)
    print(num_samples)

    vectorizer = IDVectorizer()
    vectorizer.fit(texts)

    # hyperparameters
    batch_size = 20
    embedding_dim = 128
    hidden_dim = 128
    vocab_size = len(vectorizer.vocabulary)
    print_every = 1000

    # model
    model = Decoder(vocab_size, embedding_dim, hidden_dim)
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(100):
        shuffled_idx = torch.randperm(num_samples)
        epoch_loss = 0
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
            epoch_loss += loss.item()
            if batch_idx % print_every == 0:
                print("Epoch: {}, batch: {}, loss: {}".format(epoch, batch_idx, loss))
                sample_idx = random.choice(range(X.shape[1]))
                in_ = X[:, sample_idx]
                gold_ = y[:, sample_idx]
                out_ = out[:, sample_idx]
                vocab = vectorizer.vocabulary
                print("> {}".format(idx2words(in_, vocab)))
                print("= {}".format(idx2words(gold_, vocab)))
                print("< {}".format(proba2words(out_, vocab)))
        # print("Epoch {}, loss {}".format(epoch, epoch_loss))