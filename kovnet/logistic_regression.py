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
        return F.softmax(self.linear(vec), dim=1)


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
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    print("num_parameters: {}".format([param.shape for param
                                       in model.parameters()]))

    for epoch in range(1):
        for i in range(len(texts)):
            model.zero_grad()

            input_ = [texts[i]]
            output_ = [labels[i]]
            in_vec = transformer.transform(input_)
            out_vec = torch.tensor(label_encoder.transform(output_))
            res = model.forward(in_vec)
            loss = loss_function(res, out_vec)
            print(in_vec, out_vec, res, loss)

            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    execute()
