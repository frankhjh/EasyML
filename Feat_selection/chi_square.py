# -*- coding: utf-8 -*-
# Author:hujiahao

import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2

def chi_square_score(dataset,label,feat_name):
    '''
    This function tries to compute the chi-square statistic & p-value to help us know the correlation between some spcific category feature and the label,
    which is an efficient feature selection method for classification problems. Since scikit-learn has already provide this api, here I directly use it.

    Args:
    dataset:
        The dataset contains label and the specific feature. Support Pandas DataFrame.
    label:
        Target name of classification problem. String.
    feat_name:
        The specific feature we want to do the chi-square test. String.
    
    Returns:
        the chi-square statistic & p-value. Float & Float
    '''

    category_mapp=dict()
    for uv in dataset[feat_name].unique():
        category_mapp[uv]=len(category_mapp)
    
    transformed_feat=dataset[feat_name].map(category_mapp)

    chi_square_stat,p_value=chi2(np.array(transformed_feat).reshape(-1,1),dataset[label])

    return chi_square_stat[0],p_value[0]
    


