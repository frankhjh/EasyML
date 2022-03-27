# -*- coding: utf-8 -*-
# @author:hujiahao

import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics.pairwise import pairwise_distances
from util_package import one_hot_encoding,scaler


def relief_F(dataset,label,feat_li,cate_feat_li,k=10,distance_metric='manhattan'):
    '''
    This function implements the relief-F feature selection method.

    Args:
    dataset: 
        the origin data contains label and features. pandas.DataFrame.
    label:
        the target name. String.
    feat_li:
        the list of feature names, inlcuding both continuous & category features. List of String.
    cate_feat_li:
        the list of name of category features. List of String.
    k:
        choice of the num of neighbors. int.
    
    Returns:
        the weigths for each feature (sorted). list of (feature, weight)

    '''

    # one-hot encoding for category features.
    X=dataset[feat_li]
    X=one_hot_encoding(X,cate_feat_li)
    feat_li_update=list(X) # new feature list
    
    y=dataset[label]
    

    # min-max scaler for continuous features. that is, the range [0,1] for each features.
    X=scaler(X)

    # compute pairwise distances between samples in dataset.
    distances=pairwise_distances(X,metric=distance_metric) # shape: (n_samples,n_samples)


    # init weights
    weights=np.zeros(len(feat_li_update))

    n_samples=dataset.shape[0]
    for idx in tqdm(range(n_samples)):

        near_hits=[]
        near_misses=defaultdict(list) # each class, we need a list

        self_feat=X[idx,:]
        self_label=y[idx]
        labels=y.unique().tolist() # all classes

        p_self_label=len(y[y==y[idx]])/len(y)
        
        stopper=dict() 
        for label in labels:
            stopper[label]=0 # yet we do not find the k neighbors from each class.
        
        distances_idx_sorted=list() # distances between other samples with current sample (idx)
        distances[idx,idx]=np.max(distances[idx,:]) # ignore self
        
        for i in range(n_samples):
            distances_idx_sorted.append([distances[idx,i],i,y[i]])
        distances_idx_sorted.sort(key=lambda x:x[0]) 

        # find k neighbors from each class
        for i in range(n_samples):
            if distances_idx_sorted[i][2]==y[idx]:
                if len(near_hits)<k:
                    near_hits.append(distances_idx_sorted[i][1])
                elif len(near_hits)==k:
                    stopper[y[idx]]=1
            
            else: # other classes
                other_cls=distances_idx_sorted[i][2]
                if len(near_misses[other_cls])<k:
                    near_misses[other_cls].append(distances_idx_sorted[i][1])
                elif len(near_misses[other_cls])==k:
                    stopper[other_cls]=1
            
            stop=True
            for k,v in stopper.items():
                if v!=1:
                    stop=False
            
            if stop:
                break
    
        # update weights
        near_hit_term=np.zeros(len(feat_li_update))
        for neighbor_idx in near_hits:
            near_hit_term+=np.array(abs(self_feat-X[neighbor_idx,:]))
        
        near_hit_term=near_hit_term/(k*n_samples)

        near_miss_terms=dict()
        near_miss_final=np.zeros(len(feat_li_update))
        
        for label,miss_li in near_misses.items():
            near_miss_terms[label]=np.zeros(len(feat_li_update))
            for i in miss_li:
                near_miss_terms[label]+=np.array(abs(self_feat-X[i,:]))
            
            p_label=len(y[y==label])/len(y)
            near_miss_final+=(p_label/(1-p_self_label))*near_miss_terms[label]/(k*n_samples)
        
        
        weights=weights+near_miss_final-near_hit_term

    f2w=dict()
    for f,weight in zip(feat_li_update,weights):
        f2w[f]=weight
    
    sort_f2w=sorted(f2w.items(),key=lambda x:x[1],reverse=True)

    
    return sorted_f2w

        

            


                


    

        

        













    
