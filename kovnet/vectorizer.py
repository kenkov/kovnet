#! /usr/bin/env python
# coding:utf-8

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer as CountVectorizer_
from sklearn.preprocessing import LabelEncoder
import torch


class CountVectorizer(BaseEstimator, TransformerMixin):
    """PyTorch 対応の CountVectorizer"""
    def __init__(self,
                 tokenizer=lambda x: x.split(" "),
                 ngram_range=(1, 1),
                 stop_words=None,
                 lowercase=False,
                 max_df=1.0,
                 min_df=1,
                 max_features=None
                 ):
        """
        デフォルトパラメータは、前処理は行わずにスペースで区切る処理を行う
        """
        self._params = {"tokenizer": tokenizer,
                        "ngram_range": ngram_range,
                        "stop_words": stop_words,
                        "lowercase": lowercase,
                        "max_df": max_df,
                        "min_df": min_df,
                        "max_features": max_features
                        }

    @property
    def vocabulary(self):
        return self.count_vectorizer_.vocabulary_

    def fit(self, texts):
        count_vectorizer = CountVectorizer_(**self._params)
        count_vectorizer.fit(texts)
        self.count_vectorizer_ = count_vectorizer
        return self

    def transform(self, texts):
        """
        Arg:
            texts (List[str]): 文字列のリスト。
        
        Returns:
            torch.tensor

        例：

            >>> from vectorizer import CountVectorizer
            >>> vectorizer = CountVectorizer()
            >>> vectorizer.fit(["hello world", "hello Python"])
            >>> vectorizer.transform(["hello", "hello Python"])
            tensor([[ 1.,  0.,  0.],
                    [ 1.,  1.,  0.]])
        """
        # oen-hot encoding した numpy array を作成する
        sparse_array = self.count_vectorizer_.transform(texts)
        np_array = sparse_array.toarray()

        # torch tensor に変換
        tensor = torch.tensor(np_array)
        in_vec = torch.tensor(tensor, dtype=torch.float32)
        return in_vec


class IDVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 tokenizer=lambda x: x.split(" "),
                 ngram_range=(1, 1),
                 stop_words=None,
                 lowercase=False,
                 max_df=1.0,
                 min_df=1,
                 max_features=None
                 ):
        """
        デフォルトパラメータは、前処理は行わずにスペースで区切る処理を行う
        """
        self._params = {"tokenizer": tokenizer,
                        "ngram_range": ngram_range,
                        "stop_words": stop_words,
                        "lowercase": lowercase,
                        "max_df": max_df,
                        "min_df": min_df,
                        "max_features": max_features
                        }

    @property
    def vocabulary(self):
        # vocabulary_ ではないことに注意
        return self.count_vectorizer_.vocabulary

    def fit(self, texts):
        count_vectorizer = CountVectorizer_(**self._params)  # (*args, **kwargs)
        count_vectorizer.fit(texts)

        # 語彙に <unk>, <pad>, <s>, </s> を追加する
        default_vocabulary = {"<unk>": 0,
                              "<pad>": 1,
                              "<s>":   2,
                              "</s>":  3}
        fitted_vocab = count_vectorizer.vocabulary_

        # 新しい語彙辞書を作成
        start_idx = len(default_vocabulary)
        vocabulary = {key: val for key, val in default_vocabulary.items()}
        for key in fitted_vocab:
            if key not in default_vocabulary:
                vocabulary[key] = start_idx
                start_idx += 1

        print(vocabulary)

        # 語彙を更新した count vectorizer を作成
        self.count_vectorizer_ = CountVectorizer_(vocabulary=vocabulary,
                                                  **self._params)
        return self

    def transform(self, texts, max_len=20, batch_first=False):
        """
        Arg:
            texts (List[str]): 文字列のリスト。
        
        Returns:
            torch.tensor

        例：

            >>> vec = IDVectorizer()
            >>> vec.fit(["今日 は 疲れた", "明日 は 晴れる"])
            >>> vec.transform(["明後日 は 晴れる", "今日 は 元気"], max_len=5)
            tensor([[ 0,  4],
                    [ 6,  0],
                    [ 1,  1],
                    [ 1,  1],
                    [ 1,  1]], dtype=torch.int32)
        """
        # oen-hot encoding した numpy array を作成する
        tokenizer = self.count_vectorizer_.build_tokenizer()
        word2id = self.count_vectorizer_.vocabulary

        def tokenize(text):
            tokens = tokenizer(text)
            tokens = tokens[:max_len] + ["<pad>"] * (max_len - len(tokens))
            return tokens

        # ベクトル化
        vec = [[word2id.get(word, word2id["<unk>"])
                for word in tokenize(text)]
               for text in texts]
        in_vec = torch.tensor(vec, dtype=torch.int32)

        if batch_first:
            return in_vec
        
        return in_vec.t()


if __name__ == "__main__":
    vec = IDVectorizer(max_features=3)
    res = vec.fit(["今日 は 疲れた", "明日 は 晴れる"])
    res = vec.transform(["<s> 明後日 は 晴れる </s>", "<s> 今日 元気 </s>"],
                        max_len=10)
    print(res)