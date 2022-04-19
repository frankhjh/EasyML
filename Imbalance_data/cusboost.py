# -*- coding:utf-8 -*-
"""
Created on 2022-04-09 14:07:08 Sat.
@author: hujiahao
"""

import random
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

class Cus_boosting:
    '''
    The full name of this technique is <cluster-based under sampling with boosting>, it combines the
    1.cluster-based sampling technique 
    and
    2.adaboost algorithm
    to solve the binary classification problem with imbalanced data

    Original paper <CUSBoost:Cluster-based Under-sampling with Boosting for Imbalanced Classification>
    https://arxiv.org/pdf/1712.04356.pdf
    '''

    def __init__(self,data_size,sampling_ratio,num_clusters,num_estimators,base_estimator_para='default'):
        self.data_size=data_size # size of complete dataset
        self.w=np.array([1/data_size]*data_size) # initial weight for each sample (equal)

        self.sampling_ratio=sampling_ratio # sampling ratio from each cluster
        self.num_clusters=num_clusters # the number of clusters of majoity class
        self.num_estimators=num_estimators # the number of base estimator used when doing adaboost ensembling.
        self.base_estimator_para={'max_depth':5,'random_state':1} if base_estimator_para=='default' else base_estimator_para # pass the parameters for base estimator.

        self.estimators=[self.__build_base_estimator() for _ in range(num_estimators)]

    def __load_data(self):
        pass

    
    def __cluster_sampling(self):
        pass

    def __build_base_estimator(self):
        base_clf=DecisionTreeClassifier()
        base_clf.set_params(**self.base_estimator_para)

        return base_clf
    


    




