#! /usr/bin/env python
# coding:utf-8


import numpy as np


def sort_by_length(lengths, *args):
    sorted_idx = np.argsort(lengths)[::-1]

    sorted_lengths = [lengths[idx] for idx in sorted_idx]
    lst = [sorted_lengths]

    for item in args:
        sorted_item = [item[idx] for idx in sorted_idx]
        lst.append(sorted_item)
    return lst
