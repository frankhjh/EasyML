# -*- coding:utf-8 -*-
"""
Created on 2022-04-18 17:15 Monday
@author: hujiahao
"""
import pandas as pd
import numpy as np
import re
import random
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from .utils import euclidean_distance

class ClusterBasedUnderSampling:

    def __init__(self,feat_li,cate_feat_li,discount_ratio,cluster_method,k=10,sample_ratio=0.3):
        self.feat_li=feat_li
        self.cate_feat_li=cate_feat_li
        self.discount_ratio=discount_ratio
        self.cluster_method=cluster_method
        self.k=k
        self.sample_ratio=sample_ratio
        
        self.feat_li_=None # after one hot encoding
    
    def __one_hot_encoding(self,df):
        '''
        Args:
            df(pandas.DataFrame):the DataFrame contains all features. (n_samples,n_features)

        Returns:
            df(pandas.DataFrame):the DataFrame contains all features after one hot encoding for category features.
        '''
        for f in tqdm(self.cate_feat_li):
            for idx,unique_v in enumerate(df[f].unique()):
                df[f'{f}_{idx}']=df[f].apply(lambda x:1 if x==unique_v else 0)
            df=df.drop(columns=[f],axis=1)
        
        # store the feature list after one hot encoding
        self.feat_li_=list(df)
        return df

    def __normalize(self,df):
        '''
        Args:
            df(pandas.DataFrame): the dataframe after one hot encoding for category features. (n_samples,n_features)
        
        Returns:
            df(pandas.DataFrame): the dataframe after normalization.
        '''
        scaler=MinMaxScaler()

        scaled_X=scaler.fit_transform(df.astype(float))
        scaled_df=pd.DataFrame(scaled_X)
        scaled_df.columns=self.feat_li_
        
        return scaled_df

    def __cate_discounter(self,df):
        '''
        Q: Why I do this?
        A: Because when we calculate the distance of 2 samples(vectors), the category features with only 0 or 1 will have strong
        advantage compare to continuous features which have already scaled into interval [0,1]. In order to reduce this effect 
        as possible, when I calculate the distance in each category feature, i will multiply this distance with a discount ratio. 
        In practice,given the common euclidean distance calculation method, for each category feature, i will multiply them by ratio
        equals sqrt(discount_ratio)
        
        
        Args:
            df(pandas.DataFrame): the feature dataframe after normalization. (n_samples,n_features)
        
        Returns:
            the feature dataframe after category features discount process
        '''
        cate_feat_pattern=re.compile(r'[a-zA-Z_]+_[0-9]+')
        for f in list(df):
            if f==cate_feat_pattern.match(f).group():
                df[f]=df[f].apply(lambda x:np.sqrt(self.discount_ratio)*x)
        return df
    

    def __init_cluster_alg(self):
        if self.cluster_method=='kmeans':
            cluster_alg=KMeans(n_clusters=self.k,algorithm='full')
        
        return cluster_alg
    
    def __find_best_k(self): # use elbow rule
        pass 
    
    def run(self,df):

        df=self.__one_hot_encoding(df) # one hot encoding
        df=self.__normalize(df) # normalization
        df=self.__cate_discounter(df) # category features discount

        mat=df.values

        cluster_alg=self.__init_cluster_alg() # init cluster algorithm
        cluster_alg.fit(mat) # run the cluster algorithm

        clusters=defaultdict(list) # init dict to store the cluster info for each sample

        # collect the cluster info for each sample
        for i in tqdm(range(df.shape[0])):
            label=cluster_alg.labels_[i]
            clusters[label].append(i)
        
        # re-sort the samples within each cluster based on the distance between them and their centroids.
        curs=dict()
        for i in range(self.k):
            cluster=cluster_alg.predict(cluster_alg.cluster_centers_[i].reshape(1,-1))
            curs[cluster[0]]=cluster_alg.cluster_centers_[i]
        
        clusters_distances=defaultdict(dict)
        for label,sample_li in tqdm(clusters.items()):
            for idx in sample_li:
                clusters_distances[label][idx]=euclidean_distance(df.iloc[idx].values,curs[label])
            
            distance_sorted_samples=sorted(clusters_distances[label].items(),key=lambda x:x[1])
            sorted_sample_li=[i[0] for i in distance_sorted_samples]

            # replace
            clusters[label]=sorted_sample_li
        
        # sampling from each cluster
        sample_res=list()
        for cluster,samples in clusters.items():
            pass
            



        













        

    