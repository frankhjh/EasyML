# -*- coding: utf-8 -*-
# Author:hujiahao

import pandas as pd
from utils import info_entropy
from utils import info_gain



def cate_feats_corr(dataset,feat1,feat2,mehtod='arithmetic'):
    '''
    use the idea of information gain ratio to measure the correlation between 2 category features

    Args:
    dataset:
        The dataset contains 2 features. support pandas DataFrame.
    feat1:
        The first feature name. String.
    feat2:
        The second feature name. String.
    method:
        how to average the information gain from feat2 -> feat1 and feat1 -> feat2, choice:['arithmetic','geometric']
    
    Returns:
        the correlation coef_ between feat1 and feat2. Float.
    ''' 
    ig2=info_gain(dataset,feat1,feat2) # information gain brought by category feature 2
    entropy2=info_entropy(feat2) # the info entropy of category feature 2
    ig_ratio2=(ig2+1e-5)/(entropy2+1e-5)

    ig1=info_gain(dataset,feat2,feat1) # information gain brought by category feature 1
    entropy1=info_entropy(feat1) # the info entropy of category feature 1
    ig_ratio1=(ig1+1e-5)/(entropy1+1e-5)

    if method=='arithmetic':
        corr=0.5*ig_ratio2+0.5*ig_ratio1
    else: # geometric
        corr=(ig_ratio2*ig_ratio1)**0.5
    
    return corr
