#! /usr/bin/env python
# coding:utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from kovnet.vectorizer import IDVectorizer


class NGramLanguageModel(nn.Module):
    def __init__(self, vocab_size,
                 embedding_dim, context_size):
        super(NGramLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim * context_size, 256)
        self.linear2 = nn.Linear(256, vocab_size)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size

    def forward(self, vec):
        batch_size = vec.shape[0]  # batch_first == True

        input_ = self.embedding(vec).view(batch_size, -1)

        linout = self.linear1(input_)
        wordout = self.linear2(linout)

        return wordout


if __name__ == "__main__":
    texts = ["今日 は 疲れ た ので 帰っ て 来 た",
             "今日 は 雨 だ",
             "明日 は どうなる だろう"]
    num_samples = len(texts)

    # fit vectorizer
    vectorizer = IDVectorizer()
    vectorizer.fit(texts)

    # training parameters
    vocab_size = len(vectorizer.vocabulary)
    embedding_dim = 256
    context_size = 2
    batch_size = 1

    model = NGramLanguageModel(vocab_size,
                               embedding_dim,
                               context_size)
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(100):
        shuffled_idx = torch.randperm(num_samples)
        for i in range(0, num_samples, batch_size):
            X_raw = [texts[idx] for idx in shuffled_idx[i:i+batch_size]]
            max_len = 10
            X = vectorizer.transform(X_raw, batch_first=True, max_len=max_len)
            # print(X)
            for idx in range(0, max_len-context_size):
                # zero_grad
                model.zero_grad()

                # forward
                input_ = X[:,idx:idx+context_size]
                target_ = X[:,idx+context_size]
                res = model(input_)
                loss = loss_function(res, target_)

                # backward
                loss.backward()
                optimizer.step()
                print("Target: {}, loss: {}".format(target_, loss))
                # print(input_)
                # print(output_)
                # print(res)
    print(vectorizer.vocabulary)