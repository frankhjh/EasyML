# -*- coding:utf-8 -*-
'''
Created on 2022-04-14 Thursday.
@author: hujiahao
'''

import pandas as pd
import numpy as np
import random
import sys
sys.path.append('..')
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from util_package import one_hot_encoding

class Gentic_alg:
    '''
    This class contains all sub parts to do the feature selection with the idea of gentic algorithm.

    NOW IT JUST SUPPORTS TO SOLVE CLASSIFICATION PROBLEMS.
    '''
    def __init__(self,iteration,pop_size,c_prob,m_prob,feat_li,cate_feat_li,label,train_alg,hyperparameter='default'):
        '''
        initalize the parameters required to implement the gentic algorithm for feature selection.

        Args:
            iteration (int): Number of selection process.
            pop_size (int): Size of population.
            c_prob (float): Probability of crossover.
            m_prob (float): Probability of mutation.

            feat_li (list): list of features from which we want to select.
            cate_feat_li (list): list of category features.
            label (string): the name of target.
            train_alg (string): The training algorithm , now supports 1.lgb 2.rf
            hyperparameter (string or dict): The hyperparameter used for training.

        '''
        self.iteration=iteration
        self.pop_size=pop_size
        self.c_prob=c_prob
        self.m_prob=m_prob

        self.feat_li=feat_li
        self.cate_feat_li=cate_feat_li
        self.label=label

        self.train_alg=train_alg
        self.hyperparameter=hyperparameter

        self.pop=[]
        self.fitness_li=[] # to store the fitness of each chromosome
        self.ratio_li=[] # to store the cumulative fitness ratio until each chromosome
    
    def __prep_data1(self,df,feat_li):
        '''
        This function implements the first way of data preparation - no one hot encoding.

        Args:
            df (pandas.DataFrame): the pandas.DataFrame which containd the features and label.
            feat_li (list): the selected features.
        
        Returns:
            X (pandas.DataFrame): features, shape (n_instances,n_features).
            y (pandas.Series): label, shape (n_instances,).
        '''
        y=df[self.label]
        X=df[feat_li]

        for f in list(X):
            if f in self.cate_feat_li:
                X[f]=X[f].astype('category')
        return X,y
    
    def __prep_data2(self,df,feat_li,cate_feat_li):
        '''
        This function implements the second way of data preparation - one hot encoding.

        Args:
            df (pandas.DataFrame): the pandas.DataFrame which containd the features and label.
            feat_li (list): the selected features.
            cate_feat_li (list): the list of names of category features.
        
        Returns:
            X (pandas.DataFrame): features, shape (n_instances,n_features).
            y (pandas.Series): label, shape (n_instances,).
        '''
        y=df[self.label]
        X=df[feat_li]

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
    
    def geneEncoding(self):
        '''
        Randomly create fixed number of chromosomes, each of them has same length.
        '''
        chrom_length=len(self.feat_li) # the length of chromosome = the number of features
        
        num=0
        while num<self.pop_size:
            temp=[]
            has_1=False

            for i in range(chrom_length):
                rand=random.randint(0,1) # that is, for each feature, we randomly choose add (1) or not add (0)
                if rand==1:
                    has_1=True
                temp.append(rand)

            if has_1: # must make sure that each chromosome has some features (some positions are 1)
                num+=1
                self.pop.append(temp)
    
    def calFitness(self,df):
        '''
        calculate the fitness of each chromosome
        '''
        self.fitness_li.clear()

        
        for i in range(len(self.pop)):
            selected_feats=[self.feat_li[j] for j in range(len(self.feat_li)) if self.pop[i][j]==1]
            
            if self.train_alg=='lgb':
                X,y=self.__prep_data1(df,selected_feats) # data
            
            if self.train_alg=='rf':
                cate_feat_li=[]
                for f in selected_feats:
                    if f in self.cate_feat_li:
                        cate_feat_li.append(f)

                X,y=self.__prep_data2(df,selected_feats,cate_feat_li) # data
            
            clf=self.__init_model()
            fitness=cross_val_score(clf,X,y,cv=5).mean()
            self.fitness_li.append(fitness)
    
    
    def __getRatio(self):
        self.ratio_li.clear()

        # get the sum of fitness
        sum_fitness=sum(self.fitness_li)
        
        self.ratio_li.append(self.fitness_li[0]/sum_fitness)

        for i in range(1,len(self.pop)):
            cum_ratio=self.ratio_li[i-1]+self.fitness_li[i]/sum_fitness
            self.ratio_li.append(cum_ratio)
        
    def select(self):

        # calculate the ratio
        self.__getRatio()

        rand_ratio_li=list()
        for i in range(len(self.pop)):
            rand_ratio=random.random()
            random_ratio.append(rand_ratio)
        rand_ratio_li.sort()

        new_pop=list()
        i=0 
        j=0

        while i<len(self.pop):
            if rand_ratio_li[i]<self.ratio_li[j]: # random number within the range of j th chromosome
                new_pop.append(self.pop[j])
                i+=1
            else:
                j+=1
        
        # update pop
        self.pop=new_pop
    
    def crossover(self):
        for i in range(len(self.pop)-1):
            if random.random()<self.c_prob: # crossover probability
                c_point=random.randint(0,len(self.feat_li)-1) 

                tmp1,tmp2=list(),list()
                tmp1.extend(self.pop[i][:c_point])
                tmp1.extend(self.pop[i+1][c_point:])

                tmp2.extend(self.pop[i+1][:c_point])
                tmp2.extend(self.pop[i][c_point:])

                self.pop[i]=tmp1
                self.pop[i+1]=tmp2
    
    def mutation(self):
        for i in range(len(self.pop)):
            if random.random()<self.m_prob: # mutation probability
                m_point=random.randint(0,len(self.feat_li)-1)

                if self.pop[i][m_point]==1:
                    self.pop[i][m_point]=0
                else:
                    self.pop[i][m_point]=1
    
    def __getBest(self):
        best_chrom=self.pop[0]
        best_fitness=self.fitness_li[0]

        for i in range(1,len(self.pop)):
            if self.fitness_li[i]>best_fitness:
                best_fitness=self.fitness_li[i]
                best_chrom=self.pop[i]
        
        best_feats=[self.feat_li[i] for i in range(len(self.feat_li)) if best_chrom[i]==1]
        return best_feats,best_fitness
    
    
    def main(self,data_path):
        
        logs=list()
        try:
            df=pd.read_csv(data_path,usecols=[self.label]+self.feat_li)
        except:
            raise Exception('Filetype Error')
        
        # init 
        self.geneEncoding()

        for i in tqdm(range(self.iteration)): # iter
            self.calFitness(df)

            best_feats,best_fitness=self.__getBest()
            print('at iter {}, the best feature combination:{}, best fitness:{}'.format(i,best_feats,best_fitness))
            logs.append([i,best_feats,best_fitness])

            self.select()
            self.crossover()
            self.mutation()
        
        return logs









            

            















    





