# -*- coding:utf-8 -*-
# @author:hujiahao

import pandas as pd
import numpy as np
import random
import sys
sys.path.append('..')
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from util_package import one_hot_encoding

class Gentic_alg:
    '''
    This class contains all sub parts to do the feature selection with the idea of gentic algorithm.

    NOW IT JUST SUPPORTS TO SOLVE CLASSIFICATION PROBLEMS.
    '''
    def __init__(self,iteration,pop_size,c_prob,m_prob,train_alg,hyperparameter='default'):
        '''
        initalize the parameters required to implement the gentic algorithm for feature selection.

        Args:
            iteration (int): Number of selection process.
            pop_size (int): Size of population.
            c_prob (float): Probability of crossover.
            m_prob (float): Probability of mutation.
            train_alg (string): The training algorithm , now supports 1.lgb 2.rf
            hyperparameter (string or dict): The hyperparameter used for training. 

        '''
        self.iteration=iteration
        self.pop_size=pop_size
        self.c_prob=c_prob
        self.m_prob=m_prob

        self.train_alg=train_alg
        self.hyperparameter=hyperparameter
    
    def __prep_data1(self,df,label,cate_feat_li):
        '''
        This function implements the first way of data preparation - no one hot encoding.

        Args:
            df (pandas.DataFrame): the pandas.DataFrame which containd the features and label.
            label (string): the name of label.
            cate_feat_li (list): the list of names of category features.
        
        Returns:
            X (pandas.DataFrame): features, shape (n_instances,n_features).
            y (pandas.Series): label, shape (n_instances,).
        '''
        y=df[label]
        X=df.drop(columns=[label],axis=1)

        for f in list(X):
            if f in cate_feat_li:
                X[f]=X[f].astype('category')
        return X,y
    
    def __prep_data2(self,df,label,cate_feat_li):
        '''
        This function implements the second way of data preparation - one hot encoding.

        Args:
            df (pandas.DataFrame): the pandas.DataFrame which containd the features and label.
            label (string): the name of label.
            cate_feat_li (list): the list of names of category features.
        
        Returns:
            X (pandas.DataFrame): features, shape (n_instances,n_features).
            y (pandas.Series): label, shape (n_instances,).
        '''
        y=df[label]
        X=df.drop(columns=[label],axis=1)

        X=one_hot_encoding(X,cate_feat_li)
        return X,y
    
    def __init_model(self):
        if self.train_alg=='lgb':
            clf=lgb.LGBMClassifier()
            if self.hyperparameter=='default':
                clf.set_params(**{'num_leaves':20,'max_depth':5,'subsample':0.9,'reg_alpha':50.0,'reg_lambda':50.0,'random_state':1,
                               'is_unbalance':True,'verbose':-1})
            else:
                clf.set_params(**self.hyperparameter)
        
        elif self.train_alg=='rf':
            clf=RandomForestClassifier()
            if self.hyperparameter=='default':
                clf.set_params(**{'max_depth':5,'random_state':1,'class_weight':'balanced'})
            else:
                clf.set_params(**self.hyperparameter)

        return clf



    





