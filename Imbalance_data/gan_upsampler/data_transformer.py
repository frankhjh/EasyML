from sklearn.mixture import BayesianGaussianMixture
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

class D_Transformer():


    def __init__(self, train_data, categorical_cols=[], mixed_dict={}, n_clusters=5, threshold=0.005):
        
        self.meta = None
        self.train_data = train_data
        self.categorical_columns= categorical_cols
        self.mixed_columns= mixed_dict
        self.n_clusters = n_clusters
        self.threshold = threshold
        self.ordering = []
        self.output_info = []
        self.output_dim = 0
        self.components = []
        self.is_cont = []
        
        self.meta = self.get_metadata()
    
    def get_metadata(self):
        '''
        获取table中每个特征列的基本属性信息
        '''
        
        meta = []
    
        for index in range(self.train_data.shape[1]):
            column = self.train_data.iloc[:,index]
            # 类别型特征
            if index in self.categorical_columns:
                mapper = column.value_counts().index.tolist()
                meta.append({
                        "name": index,
                        "type": "categorical",
                        "size": len(mapper),
                        "i2s": mapper
                })
            # 混合类型(包含缺失值)
            elif index in self.mixed_columns.keys():
                meta.append({
                    "name": index,
                    "type": "mixed",
                    "min": column.min(),
                    "max": column.max(),
                    "modal": self.mixed_columns[index]
                })
            # 连续型特征
            else:
                meta.append({
                    "name": index,
                    "type": "continuous",
                    "min": column.min(),
                    "max": column.max(),
                })            

        return meta
    
    def fit(self):

        data = self.train_data.values

        # 保存每一个训练的bgm模型
        models = []

        for idx,info in enumerate(self.meta):
            if info['type'] == 'continuous':
                bgm = BayesianGaussianMixture(
                    self.n_clusters,
                    weight_concentration_prior_type='dirichlet_process',
                    weight_concentration_prior=0.001,
                    max_iter=100,n_init=1,random_state=100)
                
                bgm.fit(data[:,idx].reshape[-1,1])
                models.append(bgm)

                mode_weights_above_threshold = bgm.weights_ > self.threshold
                mode_freq = (pd.Series(bgm.predict(data[:,idx].reshape([-1,1]))).value_counts().keys())

                modes = []
                for i in range(self.n_clusters):
                    if (i in mode_freq) & mode_weights_above_threshold[i]:
                        modes.append(True)
                    else:
                        modes.append(False)
                self.components.append(modes)
                self.output_info += [(1,'tanh'),(np.sum(modes),'softmax')]
                self.output_dim += 1 + np.sum(modes)
            
            elif info['type'] == 'mixed':
                bgm = BayesianGaussianMixture(
                    self.n_clusters,
                    weight_concentration_prior_type='dirichlet_process',
                    weight_concentration_prior=0.001, max_iter=100,
                    n_init=1,random_state=100)

                is_cont = []
                for ele in data[:,idx]:
                    if ele not in info['modal']:
                        is_cont.append(True)
                    else:
                        is_cont.append(False)
                self.is_cont.append(is_cont)


                bgm.fit(data[:,idx][is_cont].reshape([-1,1]))

                models.append(bgm)

                mode_weights_above_threshold = bgm.weights_ > self.threshold
                mode_freq = (pd.Series(bgm.predict(data[:,idx][is_cont].reshape([-1,1]))).value_counts().keys())
                modes = []

                for i in range(self.n_clusters):
                    if (i in mode_freq) & mode_weights_above_threshold[i]:
                        modes.append(True)
                    else:
                        modes.append(False)
                
                self.components.append(modes)

                self.output_info += [(1,'tanh'),(np.sum(modes) + len(info['modal']),'softmax')]
                self.output_dim += 1 + np.sum(modes) + len(info['modal'])
            
            else: # categorial
                model.append(None)
                self.components.append(None)
                self.output_info += [(info['size'],'softmax')]
                self.output_dim += info['size']
        
        self.models = models

    def transform(self,data):

        outputs = []

        mixed_counter = 0

        for idx,info in enumerate(self.meta):
            whereiam = data[:,idx]

            if info['type'] == 'continuous':
                whereiam = whereiam.reshape([-1,1])

                # 获取每个高斯的均值
                means = self.model[idx].means_.reshape((1,self.n_clusters))
                # 获取每个高斯的标准差
                stds =  np.sqrt(self.model[idx].covariances_).reshape((1,self.n_clusters))

                features = np.empty(shape=(len(whereiam),self.n_clusters))
                # 标准化(broadcasting)
                features = (whereiam - means) / (4 * stds)

                # 候选modes
                num_opts = sum(self.components[idx])
                
                mode_select = np.zeros(len(data),dtype='int')
                # 计算每条样本属于每个候选高斯分布的概率
                probs = self.model[idx].predict_proba(whereiam.reshape([-1,1]))
                # 剔除低贡献分布
                probs = probs[:,self.components[idx]]
                
                # 为每条样本选择最匹配的mode
                for i in range(len(data)):
                    prob = probs[i] + 1e-5
                    prob = prob / sum(prob)

                    mode_select[i] = np.random.choice(np.arange(num_opts),p=prob)
                
                # one-hot encoding
                probs_onehot = np.zeros_like(probs)
                probs_onehot[np.arange(len(probs)),mode_select] = 1

                # 获取对应mode的标准化值
                index = np.arange((len(features)))
                features = features[:,self.components[idx]]
                features = features[index,mode_select].reshape([-1,1])
                features = np.clip(features,-0.99,0.99)

                # re-order
                ordered_probs_onehot = np.zeros_like(probs_onehot)
                col_sums = probs_onehot.sum(axis=0)

                n = probs_onehot.shape[1]
                largest_indices = np.argsort(-1*col_sums)[:n]

                for index_,val in enumerate(largest_indices):
                    ordered_probs_onehot[:,index_] = probs_onehot[:,val]
                
                self.ordering.append(largest_indices)

                # 存储编码
                outputs += [features,ordered_probs_onehot]

            elif info['type'] == 'mixed':
                
                whereiam = whereiam.reshape([-1,1])
                is_cont = self.is_cont[mixed_counter] # 获取该mixed特征的连续位置

                whereiam = whereiam[is_cont]

                # 获取每个高斯的均值
                means = self.model[idx].means_.reshape((1,self.n_clusters))
                # 获取每个高斯的标准差
                stds =  np.sqrt(self.model[idx].covariances_).reshape((1,self.n_clusters))

                features = np.empty(shape=(len(whereiam),self.n_clusters))
                # 标准化(broadcasting)
                features = (whereiam - means) / (4 * stds)

                # 候选modes
                num_opts = sum(self.components[idx])
                
                mode_select = np.zeros(len(data),dtype='int')
                # 计算每条样本属于每个候选高斯分布的概率
                probs = self.model[idx].predict_proba(whereiam.reshape([-1,1]))
                # 剔除低贡献分布
                probs = probs[:,self.components[idx]]
                
                # 为每条样本选择最匹配的mode
                for i in range(len(data)):
                    prob = probs[i] + 1e-5
                    prob = prob / sum(prob)

                    mode_select[i] = np.random.choice(np.arange(num_opts),p=prob)
                
                # one-hot encoding
                probs_onehot = np.zeros_like(probs)
                probs_onehot[np.arange(len(probs)),mode_select] = 1

                # 获取对应mode的标准化值
                index = np.arange((len(features)))
                features = features[:,self.components[idx]]
                features = features[index,mode_select].reshape([-1,1])
                features = np.clip(features,-0.99,0.99)

                # 为缺失值添加一种额外的mode
                extra_mode = np.zeros(len(whereiam),len(info['modal']))
                probs_onehot_tmp = np.concatenate([extra_mode,probs_onehot])

                # 最终输出
                fin_out = np.zeros([len(data), 1 + len(info['modal']) + probs_onehot.shape[1]])

                features_curser = 0
                for index_,val in enumerate(data[:,idx]):
                    if val in info['modal']: # -99
                        cate_ = list(map(info['modal'].index, [val]))[0]
                        fin_out[index_,0] = -99  # 直接将缺失值作为feature value
                        fin_out[index_,(cate_+1)] = 1
                    else:
                        fin_out[index_,0] = features[features_curser]
                        fin_out[index_,(1 + len(info['modal'])):] = probs_onehot_tmp[features_curser][len(info['modal']):]
                        features_curser += 1
                
                just_onehot = fin_out[:,1:]
                ordered_just_onehot = np.zeros_like(just_onehot)

                n = just_onehot.shape[1]
                col_sums =just_onehot.sum(axis=0)
                largest_indices = np.argsort(-1*col_sums)[:n]

                for index_,val in enumerate(largest_indices):
                    ordered_just_onehot[:,index_] = just_onehot[:,val]
                
                fin_features = fin_out[:,0].reshape([-1,1])

                self.ordering.append(largest_indices)
                outputs += [fin_features,ordered_just_onehot]

                mixed_counter += 1
            
            else:
                # category(可直接将missing value作为一类)
                self.ordering.append(None)
                out_cols = np.zeros([len(data),info['size']])
                index = list(map(info['i2s'].index, whereiam))
                out_cols[np.arange(len(data)),index] = 1
                outputs.append(out_cols)
        
        return np.concatenate(outputs,axis=1)

    def inverse_transform(self,data):
        '''
        data: generated data by GAN
        '''   
        # 存储转换回来的表数据
        data_trans_back = np.zeros([len(data),len(self.meta)])

        step = 0 # 遍历所有原始特征

        for idx,info in enumerate(self.meta):
            if info['type'] == 'continuous':

                # 获取feature值
                feature_val = data[:,step]
                feature_val = np.clip(feature_val,-1,1)
                
                # 获取one-hot
                onehot_val = data[:,step+1:step+1+np.sum(self.components[idx])]
                
                # order back
                order = self.ordering[idx]
                onehot_val_re_order = np.zeros_like(onehot_val)    
                for index,val in enumerate(order):
                    onehot_val_re_order[:,val] = onehot_val[:,index]

                onehot_val = onehot_val_re_order

                onehot_val_t = np.ones((data.shape[0],self.n_clusters)) * -100
                onehot_val_t[:,self.components[idx]] = onehot_val

                onehot_val = onehot_val_t

                means = self.model[idx].means_.reshape([-1])
                stds = np.sqrt(self.model[idx].covariances_).reshape([-1])
                p_argmax = np.argmax(onehot_val,axis=1)
                
                std = stds[p_argmax]
                mean = means[p_argmax]

                # inverse transformation
                inverse_feature_val = feature_val * 4 * std + mean

                data_trans_back[:,idx] = inverse_feature_val

                step += 1 + np.sum(self.components[idx])
            
            elif info['type'] == 'mixed':

                feature_val = data[:,step]
                feature_val = np.clip(feature_val,-1,1)

                # 获取one-hot编码
                full_onehot_val = data[:,(step+1):(step+1)+len(info['modal'])+np.sum(self.components[idx])]

                order = self.ordering[idx]
                full_onehot_val_re_order = np.zeros_like(full_onehot_val)
                for index,val in enumerate(order):
                    full_onehot_val_re_order[:,val] = full_onehot_val[:,index]
                
                full_onehot_val = full_onehot_val_re_order

                # modes of categorical value ---> -99 
                mixed_onehot_val = full_onehot_val[:,:len(info['modal'])] # 位于最前
                # modes of continuous value
                continuous_onehot_val = full_onehot_val[:,-np.sum(self.components[idx]):]

                onehot_val_t = np.ones((data.shape[0],self.n_clusters)) * -100
                onehot_val_t[:,self.components[idx]] = continuous_onehot_val


                final_onehot_val = np.concatenate([mixed_onehot_val,onehot_val_t],axis=1)
                p_argmax = np.argmax(final_onehot_val,axis=1)

                # 对于continuous变量计算
                means = self.model[idx].means_.reshape([-1])
                stds = np.sqrt(self.model[idx].covariances_).reshape([-1])

                result =np.zeros_like(feature_val)

                for index in range(len(data)):
                    if p_argmax[index] < len(info['modal']):
                        argmax_val = p_argmax[index]
                        result[index] = float(list(map(info['modal'].__getitem__,[argmax_val]))[0])
                    else:
                        std = stds[(p_argmax[index] - len(info['modal']))]
                        mean = means[(p_argmax[index] - len(info['modal']))]
                        result[index] = feature_val[index] * 4 * std + mean
                
                data_trans_back[:,idx] = result

                step += 1 + np.sum(self.components[idx]) + len(info['modal'])

            else: # pure category

                cate_onehot = data[:,step:step + info['size']]
                index = np.argmax(cate_onehot,axis=1)

                data_trans_back[:,idx] = list(map(info['i2s'].__getitem__,index))
                step += info['size']
        
        return data_trans_back

class Img_Transformer():

    def __init__(self,side):
        self.height = side
    
    def transform(self,data):

        if self.height * self.height > len(data[0]):
            # 表格特征维度不够
            padding = torch.zeros((len(data),self.height * self.height - len(data[0]))).to(data.device)
            data = torch.cat([data,padding],axis=1)
        
        return data.view(-1,1,self.height,self.height)
    
    def inverse_transform(self,data):
        data = data.view(-1,self.height * self.height)

        return data





                
            






















                




                



















                











                



















                














    

    
