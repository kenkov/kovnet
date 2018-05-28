#! /usr/bin/env python
# coding:utf-8

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer as CountVectorizer_
from sklearn.preprocessing import LabelEncoder
import torch


class CountVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._count_vectorizer = CountVectorizer_()  # (*args, **kwargs)

    def fit(self, texts):
        return self._count_vectorizer.fit(texts)

    def transform(self, texts):
        """
        Arg:
            texts (List[str]): 文字列のリスト。
        
        Returns:
            torch.tensor

        例：

            >>> transformer = CountVectorizer()
        """
        # oen-hot encoding した numpy array を作成する
        sparse_array = self._count_vectorizer.transform(texts)
        np_array = sparse_array.toarray()

        # torch tensor に変換
        tensor = torch.tensor(np_array)
        in_vec = torch.tensor(tensor, dtype=torch.float32)
        return in_vec
