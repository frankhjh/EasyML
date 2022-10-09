# -*- coding:utf-8 -*-
"""
Created on 2022-04-18 17:15 Monday
@author: hujiahao
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import random
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import Birch
from fcmeans import FCM
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances

class ClusterBasedUnderSampling:

    def __init__(self,label,feat_li,cate_feat_li,discount_ratio,cluster_method,distance_measure,k = 10,sample_ratio = 0.3,large_cluster_threshold = 3,small_cluster_threshold = 0.1):
        self.label = label
        self.feat_li = feat_li
        self.cate_feat_li = cate_feat_li
        self.discount_ratio = discount_ratio
        self.cluster_method = cluster_method
        self.distance_measure = distance_measure
        self.k = k
        self.sample_ratio = sample_ratio
        self.large_cluster_threshold = large_cluster_threshold
        self.small_cluster_threshold = small_cluster_threshold
        
        self.feat_li_ = None # after one hot encoding
        self.cate_feat_li_ = None 
    
    def __one_hot_encoding(self,df):
        '''
        Do the one-hot encoding for category features.
        
        Args:
            df(pandas.DataFrame):the feature dataset we want to sample from. (n_instances,n_features)

        Returns:
            df(pandas.DataFrame):the feature dataset after one hot encoding for category features. (n_instances,n_features_)
        '''
        cate_feat_li_ = list()

        for f in tqdm(self.cate_feat_li):
            for idx,unique_v in enumerate(df[f].unique()):
                df[f'{f}_{idx}']=df[f].apply(lambda x:1 if x == unique_v else 0)
                cate_feat_li_.append(f'{f}_{idx}')
            df = df.drop(columns = [f],axis = 1)
        
        # store the feature list after one hot encoding
        self.feat_li_ = list(df)
        self.cate_feat_li_ = cate_feat_li_
        return self,df

    def __normalize(self,df):
        '''
        Do the normalization.
        
        Args:
            df(pandas.DataFrame): the feature dataset after one hot encoding. (n_instances,n_features_)
        
        Returns:
            scaled_df(pandas.DataFrame): the feature dataset after normalization.(n_instances,n_features_)
        '''
        scaler = MinMaxScaler()

        scaled_X = scaler.fit_transform(df.astype(float))
        scaled_df = pd.DataFrame(scaled_X)
        scaled_df.columns = self.feat_li_
        
        return scaled_df

    def __cate_discounter(self,df):
        '''
        Q: Why I do this?
        A: Because when we calculate the distance of 2 samples(vectors), the category features with only 0 or 1 will have strong
        advantage compare to continuous features which have already scaled into interval [0,1]. In order to reduce this effect 
        as possible, when I calculate the distance in each category feature, i will multiply this distance with a discount ratio. 
        In practice,given the common euclidean distance calculation method, for each category feature, i will multiply them by ratio
        equals sqrt(discount_ratio)
        
        
        Args:
            df(pandas.DataFrame): the feature dataset after normalization. (n_instances,n_features_)
        
        Returns:
            df(pandas.DataFrame): the feature dataset after category features discount process. (n_instances,n_features_)
        '''
        continuous_feat_li = list(set(self.feat_li).difference(set(self.cate_feat_li)))
        tmp = df[continuous_feat_li]
        dr = tmp.std().mean() # avg std of continous features (after normalization) 

        cate_feat_pattern = re.compile(r'[a-zA-Z_]+_[0-9]+')
        for f in list(df):
            if cate_feat_pattern.match(f):
                if f == cate_feat_pattern.match(f).group():
                    if self.discount_ratio: 
                        df[f] = df[f].apply(lambda x:np.sqrt(self.discount_ratio)*x)
                    else: # use the avg std of continuous features (after normalization) as discount rate
                        df[f] = df[f].apply(lambda x:np.sqrt(dr)*x)
        
        return df
    
    def __init_cluster_alg(self):
        '''
        Initialization of cluster algorithm
        '''
        if self.cluster_method in ['kmeans','fcmeans','birch','minikmeans']:
            if self.cluster_method == 'kmeans':
                cluster_alg = KMeans(n_clusters = self.k,algorithm = 'full')
            
            if self.cluster_method == 'fcmeans':
                cluster_alg = FCM(n_clusters = self.k)
            
            if self.cluster_method == 'birch':
                cluster_alg = Birch(n_clusters = self.k)
            
            if self.cluster_method == 'minikmeans':
                cluster_alg = MiniBatchKMeans(n_clusters = self.k,batch_size = 4096)
        else:
            raise Exception('4 cluster algorithms available now:1.kmeans 2.fcmeans 3.birch 4.minikmeans.')
        
        return cluster_alg
    
    def find_best_k(self,df,candidate_ks,method = 'elbow'): # use elbow rule
        '''
        When you use kmeans as your cluster algorithm, you may want to select the best k.

        Args:
            df(pandas.DataFrame): the feature dataset. (n_instances,n_features)
            candidates_ks(list): the list of candidate k.
            method(str): the method to select best k.
        
        Returns:
            self
        '''
        tmp = df.copy()
        self,tmp = self.__one_hot_encoding(tmp) # one hot encoding
        tmp = self.__normalize(tmp) # normalization
        tmp = self.__cate_discounter(tmp) # category features discount
        
        mat = tmp.values

        res = dict() # create a dictionary to store the evaluation result for each k.

        if method == 'elbow':
            for k in candidate_ks:
                alg=KMeans(n_clusters = k,algorithm = 'full')
                alg.fit(mat)
                
                res[k] = alg.inertia_
            plt.plot(list(res.keys()),list(res.values()))
            plt.xlabel('k')
            plt.ylabel('SSD')
            plt.title('SSD variation with different k')
            
        if method == 'silhouette':
            for k in candidate_ks:
                alg = KMeans(n_clusters = k,algorithm = 'full')
                alg.fit(mat)

                labels = alg.labels_

                res[k] = silhouette_score(mat,labels)
            plt.plot(list(res.keys()),list(res.values()))
            plt.xlabel('k')
            plt.ylabel('SC')
            plt.title('Silhouette Cofficient variation with different k')
        
        return self
    
    def run_cluster_alg(self,df):
        '''
        Run the cluster algorithm.

        Args:
            df(pandas.DataFrame): the feature dataset. (n_instances,n_features)
        
        Returns:
            the fitted model.
        '''
        tmp = df.copy() # make copy of data
        cluster_alg = self.__init_cluster_alg() # init cluster algorithm
        
        if self.cluster_method == 'kmeans':
            self,tmp = self.__one_hot_encoding(tmp) # one hot encoding
            tmp = self.__normalize(tmp) # normalization
            tmp = self.__cate_discounter(tmp) # category features discount
            mat = tmp.values
            
            cluster_alg.fit(mat) # run the cluster algorithm

        if self.cluster_method == 'fcmeans':
            self,tmp = self.__one_hot_encoding(tmp) 
            tmp = self.__normalize(tmp) 
            tmp = self.__cate_discounter(tmp) 
            mat = tmp.values
            
            cluster_alg.fit(mat) 
        
        if self.cluster_method == 'birch':
            self,tmp = self.__one_hot_encoding(tmp)
            tmp = self.__normalize(tmp)
            tmp = self.__cate_discounter(tmp)
            mat = tmp.values

            cluster_alg.fit(mat)
        
        if self.cluster_method == 'minikmeans':
            self,tmp = self.__one_hot_encoding(tmp)
            tmp = self.__normalize(tmp)
            tmp = self.__cate_discounter(tmp)
            mat = tmp.values

            cluster_alg.fit(mat)
        
        return cluster_alg,tmp
    
    def get_label(self,cluster_model,df):
        '''
        Get the label for each instance based on the trained cluster model

        Args:
            cluster_model: the fitted model.
            df(pandas.DataFrame): the feature dataset we use for cluster algorithm training.(n_instances,n_features_)
        
        Returns:
            clusters(dict): the dict which store the cluster label for each instance. 
            the key of the dict is the cluster label. e.g. 0
            the value of the dict is the list of instance index. e.g. [2,4,6,8,10] 

            Example: {0:[2,4,6,8,10],1:[1,3,5,7,9],...}
        '''
        clusters = defaultdict(list) # create the dictionary to store the cluster info for each instance

        cluster_alg = cluster_model 
        tmp = df.copy()
        
        # collect the label for each instance in dataset
        for i in tqdm(range(tmp.shape[0])):
            if self.cluster_method in ['kmeans','birch','minikmeans']: 
                label=cluster_alg.labels_[i]
            if self.cluster_method == 'fcmeans':
                label=cluster_alg.predict(tmp.iloc[i].values.reshape(1,-1))
                label=int(label[0])
            clusters[label].append(i)
        
        # re-sort the samples within each cluster based on the distance between them and their centroids.
        curs = dict()
        
        if self.cluster_method == 'kmeans' or self.cluster_method == 'minikmeans':
            for i in range(self.k):
                cluster = cluster_alg.predict(cluster_alg.cluster_centers_[i].reshape(1,-1))
                curs[cluster[0]] = cluster_alg.cluster_centers_[i]
            
        if self.cluster_method == 'fcmeans':
            for i in range(len(cluster_alg.centers)):
                cluster = cluster_alg.predict(cluster_alg.centers[i].reshape(1,-1))
                curs[int(cluster[0])] = cluster_alg.centers[i]

        if self.cluster_method == 'birch':
            for label,idxs in clusters.items():
                cur_est = list()
                sub_tmp = tmp.loc[idxs].reset_index(drop = True)

                for f in list(sub_tmp):
                    cur_est.append(sub_tmp[f].mean())
                curs[label] = np.array(cur_est)

        
        clusters_distances = defaultdict(dict)
        for label,sample_li in tqdm(clusters.items()):
            for idx in sample_li:
                if self.distance_measure == 'cos':
                    clusters_distances[label][idx] = cosine_similarity(tmp.iloc[idx].values.reshape(1,-1),curs[label].reshape(1,-1)).reshape(-1)[0]
                else: # default 欧式空间
                    clusters_distances[label][idx] = euclidean_distance(tmp.iloc[idx].values.reshape(1,-1),curs[label].reshape(1,-1)).reshape(-1)[0]
            
            distance_sorted_samples = sorted(clusters_distances[label].items(),key = lambda x:x[1])
            sorted_sample_li = [i[0] for i in distance_sorted_samples]

            # replace
            clusters[label] = sorted_sample_li
        
        return clusters

    def sample(self,clusters,df):
        '''
        Do the sampling based on the selected cluster algorithm.

        Args:
            cluster_labels(dict): the dict which stores the cluster label for each instance in the dataset.
            df(pandas.DataFrame): the dataset contains both label and raw features.(n_instances,n_features + 1)
            
        Returns:
            sample_df(pandas.DataFrame): the sampled dataset. (n_samples,n_features + 1)
        '''
        tmp = df.copy()

        # sampling from each cluster
        sample_res = list()

        avg_cluster_size = int(tmp.shape[0]/self.k)
        
        for cluster,samples in clusters.items():
            if len(samples)>avg_cluster_size*self.large_cluster_threshold:
                sampling_size = avg_cluster_size
            elif len(samples)<avg_cluster_size*self.small_cluster_threshold:
                sampling_size = len(samples)
            else:
                sampling_size = int(len(samples)*self.sample_ratio)
            
            sample_res += random.sample(samples,sampling_size)
        
        
        sampled_df = tmp.loc[sorted(sample_res)].reset_index(drop=True) # sample result
        
        return sampled_df
    
    def cbus_transformer(self,df):
        '''
        the complete transformer: origin dataset(contains both label and raw features) -> sampled dataset(contain both label and raw features)

        Args:
            df(pandas.DataFrame): the origin dataset we want to sample from and it should contains both label and complete feature list. (n_instances,n_features + 1)
        
        Returns:
            sampled_df(pandas.DataFrame): the sampled dataset. (n_samples,n_features + 1)
        '''
        try:
            df_label_feats = df[[self.label]+self.feat_li]
        except:
            raise Exception('Make sure the dataset contains the label and all features needed!')

        df_feats = df_label_feats[self.feat_li]
        cluster_model,temp = self.run_cluster_alg(df_feats)

        clusters = self.get_label(cluster_model,temp)

        sampled_df = self.sample(clusters,df_label_feats)

        return sampled_df




















        

    