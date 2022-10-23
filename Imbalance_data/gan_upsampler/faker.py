import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn import BCELoss, CrossEntropyLoss,SmoothL1Loss
from condloss import cond_loss
from nnet import *
from data_transformer import *
from utils import *

class Cond_vector():

    def __init__(self,data,output_info):
        '''
        data:经过重编码的表格数据
        output_info:对原始特征重编码后的相关信息 (feature value & mode type)
        '''

        self.model = list() # 存储每个独热表征下每条数据对应的index
        self.interval = list() #每个独热表征的起始位置和长度
        self.n_col = 0 # 所有的独热特征总和
        self.n_opt = 0 # 所有特征的独热编码位数总和
        self.p_log_sampling = list() #每个独热表征不同位置的log_prob
        self.p_sampling = list() #每个独热表征不同位置的prob

        start = 0
        for item in output_info:
            if item[1] == 'tanh':
                start += item[0]
                continue
            elif item[1] == 'softmax':
                end = start + item[0]
                self.model.append(np.argmax(data[:,start:end],axis=-1)) # 获取数据集在每一个独热表征上的index序列 (len(data),)
                self.interval.append((self.n_opt,item[0])) # 独热表征位置计数: (start_point,独热表征长度）

                self.n_col += 1
                self.n_opt += item[0]
                
                freq = np.sum(data[:,start:end],axis=0) # 独热表征每个位置的频次统计
                log_freq = np.log(freq + 1) # log of 频次
                log_pmf = log_freq / np.sum(log_freq) # log of 频次 比例
                self.p_log_sampling.append(log_pmf)
                
                pmf = freq / np.sum(freq) # 频次 比例
                self.p_sampling.append(pmf)

                start = end
        
        self.interval = np.asarray(self.interval)

    def train_sample(self,batch):
        '''
        在训练过程中进行cond_vec 采样
        '''

        if self.n_col == 0:
            return None
        batch = batch

        cond_vec = np.zeros((batch,self.n_opt),dtype = 'float32') # 初始化

        idx = np.random.choice(np.arange(self.n_col),batch) # randomly选择特征

        mask = np.zeros((batch,self.n_col),dtype='float32') # 记录batch中每个样本随机选择的特征 
        mask[np.arange(batch),idx] = 1

        category_val = col_based_sampling(self.p_log_sampling,idx) # 基于选定的特征(col)以及真实数据对应的独热概率分布进行抽样

        for i in np.arange(batch):
            cond_vec[i,self.interval[idx[i],0] + category_val[i]] = 1 # 选择独热编码位置
        
        return cond_vec,mask,idx,category_val
    
    def sample(self,batch):
        '''
        在训练外进行cond_vec 采样
        '''
        if self.n_col == 0:
            return None
        
        batch = batch

        cond_vec = np.zeros((batch,self.n_opt),dtype='float32')

        idx = np.random.choice(np.arange(self.n_col),batch)

        category_val = col_based_sampling(self.p_sampling,idx)

        for i in np.arange(batch):
            cond_vec[i,self.interval[idx[i],0] + category_val[i]] = 1
        
        return cond_vec

class Sampler():
    '''
    基于给定的cond vec 对真实样本进行抽样
    '''
    
    def __init__(self,data,output_info):
        super(Sampler,self).__init__()

        self.data = data # 重编码后的表格数据
        self.model = [] # 存储每个特征对应的独热表征的每个位置非0的元素index --- list of list of arrays
        self.n =len(data) # 真实数据量

        start = 0
        for item in output_info:
            if item[1] == 'tanh':
                start += item[0]
                continue
            elif item[1] == 'softmax':
                end = start + item[0]
                tmp = []
                for j in range(item[0]):
                    tmp.append(np.nonzero(data[:,start + j])[0]) # 返回每个位置非0元素的index序列

                self.model.append(tmp)
                start = end
    
    def sample(self,n,col,opt):
        '''
        n: 抽样个数
        col:选择的col列表
        opt:对应每个col下的哪个opt
        '''

        if col is None:
            idx = np.random.choice(np.arange(self.n),n) # 直接从原始数据量self.n中抽取特定数量(n)的数据
            return self.data[idx]
        
        idx = []
        for c,v in zip(col,opt):
            idx.append(np.random.choice(self.model[c][v])) # 从指定col下指定opt下的非0元素index中随机选择一个
        
        return self.data[idx]

class Faker():

    def __init__(self,
                 rand_dim = 100,
                 class_dim = (256,256,256,256),
                 num_channels = 64,
                 l2scale = 1e-5,
                 batch_size = 512,
                 epochs=1,
                 use_nn_mode = 'mlp'):
        
        self.rand_dim = rand_dim # 初始输入的噪声维度
        self.class_dim = class_dim # 分类器隐藏层神经元
        self.num_channels = num_channels # 图像数据channel数
        self.d_side = None # 判别器输入图像尺寸
        self.g_side = None # 生成器输入图像尺寸
        self.l2scale = l2scale # L2 正则化强度
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.generator = None
        self.discriminator = None
        self.classifier = None

        self.use_nn_mode = use_nn_mode
    
    def fit(self,train_data = pd.DataFrame,categorial=[],mixed={},ml_task={}):

        problem_type = None
        target_index = None

        # 识别监督学习的label
        if ml_task:
            problem_type = list(ml_task.keys())[0]
            if problem_type:
                target_index = train_data.columns.get_loc(ml_task[problem_type])
        
        # 对初始表格数据进行重编码的transformer
        self.transformer = D_Transformer(train_data=train_data,categorical_cols=categorial,mixed_dict=mixed)
        self.transformer.fit()

        # 编码转换
        train_data = self.transformer.transform(train_data.values)
        data_dim = self.transformer.output_dim

        # 定义【采样器】和【condition vector 生成器】
        data_sampler = Sampler(train_data,self.transformer.output_info)
        self.cond_generator = Cond_vector(train_data,self.transformer.output_info)

        if self.use_nn_mode == 'cnn':
            # 判别器的输入尺寸
            sides_d = [4,8,16,24,32,48]
            col_size_d = data_dim + self.cond_generator.n_opt
            for i in sides_d:
                if i**2 >= col_size_d:
                    self.d_side = i
                    break
            
            # 生成器的输入尺寸
            sides_g = [4,8,16,24,32,48]
            col_size_g = data_dim
            for i in sides_g:
                if i**2 >= col_size_g:
                    self.g_side = i
                    break
        
            # 初始化 生成器 & 判别器
            self.generator = Generator(self.g_side,self.rand_dim+self.cond_generator.n_opt,self.num_channels).to(self.device)
            self.discriminator = Discriminator(self.d_side,self.num_channels).to(self.device)

            # 定义 生成器 和 判别器的 图像数据转换器
            self.G_transformer = Img_Transformer(self.g_side)
            self.D_transformer = Img_Transformer(self.d_side)
        
        if self.use_nn_mode == 'mlp':

            self.generator = MLP_Generator(self.rand_dim+self.cond_generator.n_opt,data_dim,3).to(self.device)
            self.discriminator = MLP_Discriminator(data_dim+self.cond_generator.n_opt,3).to(self.device)

        if self.use_nn_mode == 'attn_mlp':

            self.generator = Attn_MLP_Generator(self.rand_dim+self.cond_generator.n_opt,data_dim,3).to(self.device)
            self.discriminator = Attn_MLP_Discriminator(data_dim+self.cond_generator.n_opt,3).to(self.device)
        
        # 定义训练的超参数字典
        opt_params = dict(lr=1e-5, betas=(.5,.9), eps=1e-3, weight_decay=self.l2scale)
        opt_G = Adam(self.generator.parameters(),**opt_params)
        opt_D = Adam(self.discriminator.parameters(),**opt_params) 


        # 标签列相关信息初始化
        start_end = None
        classifier = None
        opt_C = None

        if target_index: # 如果含有标签列
            start_end = get_start_end(target_index,self.transformer.output_info) # 找到标签列的起始和结束位置
            
            # 初始化 分类器
            self.classifier = Classifier(data_dim,self.class_dim,start_end).to(self.device)
            # 定义 优化器
            opt_C = Adam(self.classifier.parameters(),**opt_params)
        
        # 对 生成器 和 判别器 初始化权重
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        
        
        steps_per_epoch = max(1,len(train_data)) // self.batch_size
        for i in tqdm(range(self.epochs)):
            for _ in range(steps_per_epoch):

                noisez = torch.randn(self.batch_size,self.rand_dim,device=self.device) # 随机生成指定维度(self.rand_dim)的噪音
                cond_vec = self.cond_generator.train_sample(self.batch_size) # 生成训练时使用的cond_vec

                c,m,col,opt = cond_vec
                c = torch.from_numpy(c).to(self.device) # cond vec
                m = torch.from_numpy(m).to(self.device) # 选择的特征 col

                noisez = torch.cat([noisez,c], dim=1) # 合并 noise & cond_vec
                if self.use_nn_mode == 'cnn':
                    noisez = noisez.view(self.batch_size,self.rand_dim+self.cond_generator.n_opt,1,1) # reshape

                real_idx = np.arange(self.batch_size) # 先按序生成真实数据index
                np.random.shuffle(real_idx) # 打乱顺序
                real = data_sampler.sample(self.batch_size,col[real_idx],opt[real_idx]) # 生成batch size个真实数据
                real = torch.from_numpy(real.astype('float32')).to(self.device)

                c_real = c[real_idx] # 生成batch size个 cond vec

                # 生成器生成虚假数据
                fake = self.generator(noisez)

                if self.use_nn_mode == 'cnn':
                    # 将其转换为原始表格形式数据
                    fake_t = self.G_transformer.inverse_transform(fake)
                    # 激活函数
                    act_fake_t = apply_activate(fake_t,self.transformer.output_info)
                    
                    # concat
                    fake_cat = torch.cat([act_fake_t,c],dim=1)
                    real_cat = torch.cat([real,c_real],dim=1)

                    # 转换为判别器输入需要的图像格式
                    fake_cat_d = self.D_transformer.transform(fake_cat)
                    real_cat_d = self.D_transformer.transform(real_cat)
                
                if self.use_nn_mode == 'mlp' or self.use_nn_mode == 'attn_mlp':
                    act_fake = apply_activate(fake,self.transformer.output_info)

                    fake_cat_d = torch.cat([act_fake,c],dim=1)
                    real_cat_d = torch.cat([real,c_real],dim=1)

                ################################# Update Discriminator ##################################

                # 梯度清除
                opt_D.zero_grad()
                # 判别器正向传播
                y_real,_ = self.discriminator(real_cat_d)
                y_fake,_ = self.discriminator(fake_cat_d)

                # 计算判别器loss ---> 【max】 likelihood log(D(x)) + log(1 - D(G(z))) ---> 【min】 loss_d = -log(D(x)) - log(1-D(G(z)))
                loss_d = (-(torch.log(y_real + 1e-4).mean()) - (torch.log(1. - y_fake + 1e-4).mean()))

                loss_d.backward() # 后向传播
                opt_D.step() # 参数更新
                
                ##################################### Update Discriminator Done ###########################################

                
                
                noisez = torch.randn(self.batch_size,self.rand_dim,device=self.device)
                cond_vec = self.cond_generator.train_sample(self.batch_size)

                c,m,col,opt = cond_vec
                c = torch.from_numpy(c).to(self.device)
                m = torch.from_numpy(m).to(self.device)

                noisez = torch.cat([noisez,c], dim=1) 
                if self.use_nn_mode == 'cnn':
                    noisez = noisez.view(self.batch_size,self.rand_dim+self.cond_generator.n_opt,1,1)

                
                
                ################################### Update Generator ####################################################

                # 梯度清除
                opt_G.zero_grad()

                fake = self.generator(noisez)

                if self.use_nn_mode == 'cnn':
                    fake_t = self.G_transformer.inverse_transform(fake)
                if self.use_nn_mode == 'mlp' or self.use_nn_mode == 'attn_mlp':
                    fake_t = fake
                act_fake_t = apply_activate(fake_t,self.transformer.output_info)
                fake_cat = torch.cat([act_fake_t,c],dim=1)

                if self.use_nn_mode == 'cnn':
                    fake_cat_d = self.D_transformer.transform(fake_cat)
                if self.use_nn_mode == 'mlp' or self.use_nn_mode == 'attn_mlp':
                    fake_cat_d = fake_cat

                y_fake,info_fake = self.discriminator(fake_cat_d) # 虚假样本正向传播
                _,info_real = self.discriminator(real_cat_d) # 真实样本正向传播
                conditional_loss = cond_loss(fake_t,self.transformer.output_info,c,m) # 计算 conditional loss

                # 生成器loss part1 = condition loss + - log(D(G(z)))
                loss_g = conditional_loss - (torch.log(y_fake + 1e-5).mean())
                # 保留计算图，因为我们还需要information loss 去计算最终的梯度
                loss_g.backward(retain_graph=True) 

                # 生成器loss part2 = information loss 
                # 计算均值和方差用于评估真实样本和虚假样本的统计分布差异
                loss_mean = torch.norm(torch.mean(info_fake.view(self.batch_size,-1),dim=0) - torch.mean(info_real.view(self.batch_size,-1),dim=0),1)
                loss_std = torch.norm(torch.std(info_fake.view(self.batch_size,-1),dim=0) - torch.std(info_real.view(self.batch_size,-1),dim=0),1)

                loss_info = loss_mean + loss_std

                loss_info.backward() # 反向传播

                opt_G.step() # 最终进行参数更新


                # 分类任务损失
                if problem_type:

                    c_loss = None

                    if (start_end[1] - start_end[0]) == 2: # 2分类
                        c_loss = BCELoss()
                    else: # 多分类
                        c_loss = CrossEntropyLoss() 
                    
                    opt_C.zero_grad()
                    real_pred,real_label = self.classifier(real) 
                    if (start_end[1] - start_end[0]) == 2:
                        real_label = real_label.type_as(real_pred)
                    
                    loss_classification = c_loss(real_pred,real_label) # 真实样本训练分类器
                    loss_classification.backward()
                    opt_C.step()


                    # 分类器用于评估生成器生成样本的标签合理性
                    opt_G.zero_grad()
                    fake = self.generator(noisez)

                    if self.use_nn_mode == 'cnn':
                        fake_t = self.G_transformer.inverse_transform(fake)
                    
                    if self.use_nn_mode == 'mlp' or self.use_nn_mode == 'attn_mlp':
                        fake_t = fake

                    act_fake_t = apply_activate(fake_t,self.transformer.output_info)

                    fake_pred,fake_label = self.classifier(act_fake_t)
                    if (start_end[1] - start_end[0]) == 2:
                        fake_label = fake_label.type_as(fake_pred)
                    
                    loss_classification_g = c_loss(fake_pred,fake_label)
                    loss_classification_g.backward()
                    opt_G.step()

    def sample(self,num):
        '''
        基于训练好的生成器进行虚假样本生成
        '''
        # eval mode
        self.generator.eval()
        # col info
        output_info = self.transformer.output_info

        steps = num // self.batch_size + 1
        data = []

        for _ in range(steps):
            noisez = torch.randn(self.batch_size,self.rand_dim,device=self.device)
            cond_vec = self.cond_generator.sample(self.batch_size)
            c = cond_vec
            c = torch.from_numpy(c).to(self.device)
            noisez = torch.cat([noisez,c],dim=1)

            if self.use_nn_mode == 'cnn':
                noisez = noisez.view(self.batch_size,self.rand_dim+self.cond_generator.n_opt,1,1)
            
            fake = self.generator(noisez)

            if self.use_nn_mode == 'cnn':
                fake_t = self.G_transformer.inverse_transform(fake)
            if self.use_nn_mode == 'mlp' or self.use_nn_mode == 'attn_mlp':
                fake_t = fake
                
            act_fake_t = apply_activate(fake_t,output_info)

            data.append(act_fake_t.detach().cpu().numpy())
        
        data = np.concatenate(data,axis=0)
        result = self.transformer.inverse_transform(data)

        return result[0:num]

    























                








































    
        

        




        



    













    




    




















