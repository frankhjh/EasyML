# -*- coding: utf-8 -*-

import pandas as pd
import numpy as 

def info_entropy(feat):
    '''
    Compute the information entropy of one feature.

    Args:
    feat:
        Value list of the feature. List or pandas Series object
    
    Returns:
        The value of entropy. Float.
    '''
    num=len(feat) 
    prob_dist=pd.value_counts(feat)/num 

    return sum((-1)*prob_dist*np.log2(prob_dist))


def info_gain(dataset,feat1,feat2):
    '''
    Compute the information gain brought by feat2.

    Args:
    dataset: 
        The dataset contains 2 features. support pandas DataFrame.

    feat1:
        The first feature name. String.
    
    feat2:
        The second feature which we want to measure how much information gain it brings. String.
    
    Returns:
        The information gain. Float.
    '''
    init_entropy=info_entropy(dataset[feat1]) # entropy of the first feature

    num=len(dataset)
    cond_prob_dist=(pd.value_counts(dataset[feat2])/num).to_dict() # the distribution of the second feature

    cond_entropy=dataset.groupby(feat2).apply(lambda x:info_entropy(x[feat1])).to_dict() # the conditional entropy given each unique value of second feature

    new_entropy=0.0
    for uv in cond_prob_dist.keys():
        new_entropy+=cond_prob_dist.get(uv)*cond_entropy.get(uv)
    
    ig=init_entropy-new_entropy

    return ig








