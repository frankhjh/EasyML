# -*- coding:utf-8 -*-
"""
Created on 2022-05-25 10:40 Wednesday
@author: hujiahao
"""
import numpy as np
from scipy import optimize


class ImbalanceLoss:

    def __init__(self,**kwargs):
        pass
    
    def _grad(self,y, yhat):
        raise NotImplementedError

    def _hess(self,y, yhat):
        raise NotImplementedError
    
    def init_score(self, y):
        raise NotImplementedError

    def clip_sigmoid(self, yhat):
        raise NotImplementedError

    def obj_func(self, preds, train_data):
        raise NotImplementedError

    def eval_func(self, preds, train_data):
        raise NotImplementedError
    

class WeightedFocalLoss(ImbalanceLoss):
    
    def __init__(self, gamma, alpha):
        super(self, WeightedFocalLoss).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
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
    
    def init_score(self, y):
        res = optimize.minimize_scalar(lambda p: self(y, p).sum(), bounds=(0, 1), methods='bounded')
        p = res.x
        log_odds = np.log(p / (1 - p))
            
        return log_odds
    
    def clip_sigmoid(self, yhat):
        yhat = 1./(1 + np.exp(-yhat))
        yhat = np.clip(yhat, 1-1e-15, 1e-15)
        
        return yhat
    
    def obj_func(self, preds, train_data):
        y = train_data.get_label()
        yhat = self.clip_sigmoid(preds)

        grad = self._grad(y,yhat)
        hess = self._hess(y,yhat)
        
        return grad,hess
    
    def eval_func(self, preds, train_data):
        y = train_data.get_label()
        yhat = self.clip_sigmoid(preds)
        is_higher_better = False

        return 'weighted_focal_loss', self(y,yhat), is_higher_better


class AsymmetricFocalLoss(ImbalanceLoss):

    def __init__(self, gamma_pos, gamma_neg):
        super(self, AsymmetricFocalLoss).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
    
    def __call__(self, y, yhat):
        return -1 * np.power(1 - yhat, self.gamma_pos) * np.log(yhat) - (1 - yhat) * np.power(yhat, self.gamma_neg) * np.log(1 - yhat)
    
    def _grad(self, y, yhat):
        pt1 = y * np.power(1 - yhat, self.gamma_pos) * (self.gamma_pos * yhat * np.log(yhat) + yhat - 1)
        pt2 = -1 * (1 - yhat) * np.power(yhat, self.gamma_neg) * (self.gamma_neg * (1 - yhat) * np.log(1 - yhat) - yhat)

        return pt1 + pt2
    
    def _hess(self, y, yhat):
        pt1 = -1 * y * self.gamma_pos * np.power(1 - yhat, self.gamma_pos) * yhat * (self.gamma_pos * yhat * np.log(yhat) + yhat - 1)
        pt2 = y * np.power(1 - yhat, self.gamma_pos + 1) * yhat * (self.gamma_pos * np.log(yhat) + self.gamma_pos + 1)
        pt3 = -1 * (1 - y) * self.gamma_neg * np.power(yhat, self.gamma_neg) * (1 - yhat) * (self.gamma_neg * (1 - yhat) * np.log(1 - yhat) - yhat)
        pt4 = -1 * (1 - y) * np.power(yhat, self.gamma_neg + 1) * (1 - yhat) * (-1 * self.gamma_neg * np.log(1 - yhat) - self.gamma_neg - 1)

        return pt1 + pt2 + pt3 + pt4
    
    def init_score(self, y):
        res = optimize.minimize_scalar(lambda p: self(y, p).sum(), bounds=(0, 1), methods='bounded')
        p = res.x
        log_odds = np.log(p / (1 - p))
            
        return log_odds
    
    def clip_sigmoid(self, yhat):
        yhat = 1./(1 + np.exp(-yhat))
        yhat = np.clip(yhat, 1-1e-15, 1e-15)
        
        return yhat
    
    def obj_func(self, preds, train_data):
        y = train_data.get_label()
        yhat = self.clip_sigmoid(preds)

        grad = self._grad(y,yhat)
        hess = self._hess(y,yhat)
        
        return grad,hess
    
    def eval_func(self, preds, train_data):
        y = train_data.get_label()
        yhat = self.clip_sigmoid(preds)
        is_higher_better = False

        return 'asymmetric_focal_loss', self(y,yhat), is_higher_better
    

class DiceLoss(ImbalanceLoss):
    
    def __init__(self, gamma):
        super(self,DiceLoss).__init__()
        self.gamma = gamma
    
    def __call__(self, y, yhat):
        return 1 - ((2 * (1 - yhat) * yhat * y + self.gamma) / ((1 - yhat) * yhat + y + self.gamma))

    def _grad(self, y, yhat):
        numerator = (2 * yhat - 1) * yhat * (1 - yhat) * (2 * np.power(y, 2) + 2 * y * self.gamma + self.gamma)
        denominator = ((1 - yhat) * yhat + y + self.gamma) ** 2
        grad = numerator / denominator

        return grad
    
    def _hess(self, y, yhat):
        num_factor1 = 2 * np.power(y, 2) + 2 * y * self.gamma + self.gamma
        num_factor2 = yhat * (1 - yhat)
        num_factor3 = -6 * np.power(yhat, 2) + 6 * yhat - 1
        num_factor4 = yhat - np.power(yhat, 2) + y + self.gamma
        num_factor5 = -4 * np.power(yhat, 3) + 6 * np.power(yhat, 2) - 2 * yhat
        num_factor6 = 1 - 2 * yhat

        denominator = (yhat - np.power(yhat, 2) + y + self.gamma) ** 3

        hess = (num_factor1 * num_factor2 * (num_factor3 * num_factor4 - num_factor5 * num_factor6)) / denominator

        return hess
    
    def init_score(self, y):
        res = optimize.minimize_scalar(lambda p: self(y, p).sum(), bounds=(0, 1), methods='bounded')
        p = res.x
        log_odds = np.log(p / (1 - p))
            
        return log_odds
    
    def clip_sigmoid(self, yhat):
        yhat = 1./(1 + np.exp(-yhat))
        yhat = np.clip(yhat, 1-1e-15, 1e-15)
        
        return yhat
    
    def obj_func(self, preds, train_data):
        y = train_data.get_label()
        yhat = self.clip_sigmoid(preds)

        grad = self._grad(y,yhat)
        hess = self._hess(y,yhat)
        
        return grad,hess
    
    def eval_func(self, preds, train_data):
        y = train_data.get_label()
        yhat = self.clip_sigmoid(preds)
        is_higher_better = False

        return 'dice_loss', self(y,yhat), is_higher_better
        
    

    

    

    











    


    






    


