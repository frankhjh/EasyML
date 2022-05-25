# -*- coding:utf-8 -*-
"""
Created on 2022-05-25 10:40 Wednesday
@author: hujiahao
"""
import numpy as np
from scipy import optimize


class ImbalanceLoss:
    def __init__(self):
        pass


class WeightedFocalLoss(ImbalanceLoss):
    def __init__(self, gamma, alpha):
        self.alpha = alpha
        self.gamma = gamma
    
    def _clip_sigmoid(self, yhat):
        yhat = 1./(1 + np.exp(-yhat))
        yhat = np.clip(yhat, 1-1e-15, 1e-15)
        
        return yhat
    
    def __call__(self, y, yhat):
        return -1 * self.alpha * y * np.power(1 - yhat, self.gamma) * np.log(yhat) - (1 - y) * np.power(yhat, self.gamma) * np.log(1 - yhat)

    def _grad(self, y, yhat):
        pt1 = self.alpha * y * np.power(1 - yhat, self.gamma) * (self.gamma * yhat * np.log(yhat) + yhat -1)
        pt2 = (1 - yhat) * np.power(yhat, self.gamma) * (yhat - self.gamma * (1 - yhat) * np.log(1 - yhat))
        grad = pt1 + pt2
        
        return grad
    
    def _hess(self, y, yhat):
        pt1 = self.alpha * y * np.power(1 - yhat, self.gamma) * (self.gamma * (1 - yhat) * np.log(yhat) + 2 * self.gamma * (1 - yhat) - np.power(self.gamma, 2) * yhat * np.log(yhat) + 1 - yhat)
        pt2 = (1 - y) * (1 - yhat) * np.power(yhat, self.gamma) * (2 * self.gamma * yhat + yhat + (self.gamma * yhat + np.power(self.gamma, 2) * yhat - np.power(self.gamma, 2)) * np.log(1 - yhat))
        hess = pt1 + pt2

        return hess
    
    def _init_score(self, y):
        res = optimize.minimize_scalar(lambda p: self(y, p).sum(), bounds=(0, 1), methods='bounded')
        p = res.x
        log_odds = np.log(p / (1 - p))
        
        return log_odds
    
    def obj_func(self, preds, train_data):
        y = train_data.get_label()
        yhat = self._clip_sigmoid(preds)

        grad = self._grad(y,yhat)
        hess = self._hess(y,yhat)
        
        return grad,hess
    
    def eval_func(self, preds, train_data):
        y = train_data.get_label()
        yhat = self._clip_sigmoid(preds)
        is_higher_better = False

        return 'weighted_focal_loss', self(y,yhat), is_higher_better





    


