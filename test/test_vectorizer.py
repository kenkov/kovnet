#! /usr/bin/env python
# coding:utf-8

from kovnet.vectorizer import IDVectorizer
import unittest
import torch


class IDVectorizerTest(unittest.TestCase):
    def test_fit_transform(self):
        vec = IDVectorizer(max_features=10)
        vec.fit(["今日 は 疲れた", "明日 は 晴れる"])
        assert vec.vocabulary["<pad>"] == 0
        assert vec.vocabulary["<unk>"] == 1
        assert vec.vocabulary["<s>"] == 2
        assert vec.vocabulary["</s>"] == 3

        res = vec.transform(["<s> 明後日 は 晴れる </s>",
                             "<s> 今日 元気 </s>"],
                            max_len=10)
        print(res)
        ans = torch.tensor([[2, 2],
                            [1, 4],
                            [5, 1],
                            [8, 3],
                            [3, 0],
                            [0, 0],
                            [0, 0],
                            [0, 0],
                            [0, 0],
                            [0, 0]], dtype=torch.int32)