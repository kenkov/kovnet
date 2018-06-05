#! /usr/bin/env python
# coding:utf-8


import unittest
from kovnet.utils import sort_by_length


class SortByLengthTest(unittest.TestCase):
    def test_sort_by_length_arg1(self):
        # text samples
        lengths = [4, 5, 2, 1]
        texts = ["A B C D",
                 "A B C D E",
                 "A B",
                 "A"]
        ans_lengths = [5, 4, 2, 1]
        ans_texts = ["A B C D E",
                     "A B C D",
                     "A B",
                     "A"]

        res_lengths, res_texts = sort_by_length(lengths, texts)
        self.assertEqual(res_lengths, ans_lengths)
        self.assertEqual(res_texts, ans_texts)

    def test_sort_by_length_arg2(self):
        # text samples
        lengths = [4, 5, 2, 1]
        texts = ["A B C D",
                 "A B C D E",
                 "A B",
                 "A"]
        labels = [0, 1, 2, 3]
        ans_lengths = [5, 4, 2, 1]
        ans_texts = ["A B C D E",
                     "A B C D",
                     "A B",
                     "A"]
        ans_labels = [1, 0, 2, 3]

        res_lengths, res_texts, res_labels = sort_by_length(lengths, texts, labels)
        self.assertEqual(res_lengths, ans_lengths)
        self.assertEqual(res_texts, ans_texts)
        self.assertEqual(res_labels, ans_labels)