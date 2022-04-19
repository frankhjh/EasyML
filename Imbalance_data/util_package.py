# -*- coding: utf-8 -*-
# @author:hujiahao

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


def one_hot_encoding(X,cate_feat_li):
    '''
    This function implements the one hot encoding for category features in dataset.

    Args:
    X:
        Original features. Support pandas DataFrame.
    cate_feat_li:
        The list of name of category features. List of String.
    
    Returns:
        Transformed dataset. DataFrame.

    '''
    for f in tqdm(cate_feat_li):
        for idx,unique_v in enumerate(X[f].unique()):
            X[f'{f}_{idx}']=X[f].apply(lambda x:1 if x==unique_v else 0)
        X=X.drop(columns=[f],axis=1)
    return X


def scaler(X):
    '''
    This function implements the min-max scaler for features X. 

    Args:
        X: pandas.DataFrame ,shape:(n_samples,n_features).
    
    Returns:
        scaled X. numpy.array, shape:(n_samples,n_features).
    '''
    scaler=MinMaxScaler()
    scaled_X=scaler.fit_transform(X.astype(float))
    return scaled_X
