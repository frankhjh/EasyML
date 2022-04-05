# -*- coding:utf-8 -*-
# @author:hujiahao

import pandas as pd
import lightgbm as lgb
from tqdm import tqdm


class Stepwise_feat_selector:
    '''
    This class implements the stepwise feature selection, which is a heuristic feature selecton method, 
    it includes 2 different modes, one is forward selection, the other is backward selection.

        1.[forward selection]: in each step, the algorithm will select the feature which brings maximum 
        improvement of target function.

        2.[backward selection]: in each step, the algorithm will drop the feature without which the  
        reduction of target function is minimum.
    
    In order to overcome the drawback( easy to get into local optimization ) of these methods, here I use the 
    idea of [simulated annealing] by introducing some randomness for each step of selection.

    NOW THIS METHOD ONLY SUPPORTS TO SOLVE CLASSIFICATION PROBLEM! 
    and the default algorithm for training I use is lightgbm, which is a quite powerful boosting tree-based learning algorithm, 
    as for the hyperparameters for model training during this process, i will provide a default one,you can also change by yourself. 
    '''
    def __init__(self,
                mode='forward',
                flexible=True,
                opt_tar='ks_cost',
                train_alg='lgb',
                hyperparameters='default'):
        '''
        Initalize the mode of feature selector

        Args:
            mode (String, default:'forward'): forward or backward.
            flexible (Boolean, default:True): whether to add randomness during each step of feature selection.
            opt_tar (String, default:ks_cost): criterion of selecting (or drop) which feature in each step. ONLY SUPPORT ks_cost NOW.
            train_alg (String, default:'lgb'): the training algorithm used during each step of feature selection. ONLY SUPPORT lgb NOW.
            hyperparameter (String, default:default): the hyperparameters used for lightgbm algorithm, you should pass a string of dict if you do not want to use default parameters.
        '''
        self.mode=mode
        self.flexible=flexible
        self.opt_tar=opt_tar
        self.train_alg=train_alg
        self.hyperparameters={'num_leaves':20,'max_depth':5,'subsample':0.9,'reg_alpha':50.0,'reg_lambda':50.0,'random_state':1,
                               'is_unbalance':True,'verbose':-1} if hyperparameters=='default' else eval(hyperparameters)
    
    def init_model(self):
        '''
        Initalize the lgb classifier
        '''
        lgb_clf=lgb.LGBMClassifier()
        lgb_clf.set_params(**self.hyperparameters)

        return lgb_clf
    
    def forward_select(self):
        pass

    def backward_select(self):
        pass

    
    def runner(self):
        pass

    
