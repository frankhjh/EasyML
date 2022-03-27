# -*- coding: utf-8 -*-
# Author:hujiahao

import pandas as pd

def fisher_score_binary_clf(dataset,label,feat_name):
    '''
    This function tries to measure the correlation between the label(binary) and
    one continuous feature using the fisher score, which means how much the 
    feature distribution varies between 2 groups (labels).

    Args:
    dataset:
        The dataset contains label and the specific feature. Support Pandas DataFrame.
    label:
        Target name of classification problem, the value of target should be either 0 or 1, which represents 2 class. String.
    feat_name:
        The name of the continuous feature. String. 
    
    Returns:
        The fisher score. Float.
    '''
    feat_mean=dataset[feat_name].mean()

    pos_samples=dataset[dataset[label]==1].reset_index(drop=True) # postive samples
    neg_samples=dataset[dataset[label]==0].reset_index(drop=True) # negative samples

    pos_mean=pos_samples[feat_name].mean() # mean of postive samples
    pos_vars=pos_samples[feat_name].var() # variance of positive samples
    
    neg_mean=neg_samples[feat_name].mean() # mean of negative samples
    neg_vars=neg_samples[feat_name].var() # variance of negative samples

    fisher_score=(1e-5+(pos_mean-feat_mean)**2+(neg_mean-feat_mean)**2)/(1e-5+pos_vars+neg_vars)

    return fisher_score


def fisher_score_multiclass_clf(dataset,label,feat_name):
    '''
    This function tries to measure the correlation between the label(multi-class) and
    one continuous feature using the fisher score, which means how much the 
    feature distribution varies between different groups (labels).

    Args:
    dataset:
        The dataset contains label and the specific feature. Support Pandas DataFrame.
    label:
        Target name of classification problem. String.
    feat_name:
        The name of the continuous feature. String. 
    
    Returns:
        The fisher score. Float.
    '''
    feat_mean=dataset[feat_name].mean()

    sub_means=dataset.groupby(label).apply(lambda x:x[feat_name].mean()).to_dict() # feature means with each class
    sub_vars=dataset.groupby(label).apply(lambda x:x[feat_name].var()).to_dict() #feature variance with each class

    sum_mean_distances_squared=0.0
    sum_vars=0.0

    for clas in sub_means.keys():
        sum_mean_distances_squared+=(sub_means.get(clas)-feat_mean)**2
        sum_vars+=sub_vars.get(clas)
    
    fisher_score=(1e-5+sum_mean_distances_squared)/(1e-5+sum_vars)

    return fisher_score



