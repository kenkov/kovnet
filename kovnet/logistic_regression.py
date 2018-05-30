#! /usr/bin/env python
# coding:utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from kovnet.vectorizer import CountVectorizer
from sklearn.preprocessing import LabelEncoder


class LogisticRegression(nn.Module):
    def __init__(self, num_features, num_labels):
        super(LogisticRegression, self).__init__()

        self.linear = nn.Linear(num_features, num_labels)

    def forward(self, vec):
        return self.linear(vec)


def execute():
    texts = ["今日 は 暑い", "あした は 暑い", "少し 寒い", "寒い かも"]
    labels = [0, 0, 1, 1]

    transformer = CountVectorizer()
    transformer.fit(texts)

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    num_features = len(transformer.vocabulary)
    num_labels = len(label_encoder.classes_)

    print(num_features)
    print(num_labels)

    model = LogisticRegression(num_features, num_labels)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    print("num_parameters: {}".format([param.shape for param
                                       in model.parameters()]))
    
    batch_size = 2
    num_samples = len(texts)

    X = transformer.transform(texts)
    y = torch.tensor(label_encoder.transform(labels))

    for epoch in range(1000):
        shuffled_nums = torch.randperm(num_samples)
        for i in range(0, num_samples, batch_size):
            input_ = X[shuffled_nums[i:i+batch_size]]
            output_ = y[shuffled_nums[i:i+batch_size]]

            model.zero_grad()

            res = model.forward(input_)
            loss = loss_function(res, output_)
            # print(input_, output_, res, loss)
            print(res, output_, loss)

            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    execute()
