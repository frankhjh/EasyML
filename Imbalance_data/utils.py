# -*- coding:utf-8 -*-
"""
Created on 2022-04-19 23:33 Tuesday
@author: hujiahao
"""
import numpy as np

def euclidean_distance(a, b):
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("Missing values detected!")
    return np.sum((a - b) ** 2, axis=0)