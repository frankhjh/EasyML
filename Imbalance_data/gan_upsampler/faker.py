import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn import (Dropout, LeakyReLU, Linear, ReLU, Sequential,
Conv2d, ConvTranspose2d, BatchNorm2d, Sigmoid, init, BCELoss, CrossEntropyLoss,SmoothL1Loss)
from data_transformer import D_Transformer,Img_Transformer

def col_based_sampling(probs,col_idx):

    option_list =[]

    for i in col_idx:
        p = probs[i] + 1e-5
        p = p/sum(p)

        option_list.append(np.random.choice(np.arange(len(probs[i])),p=p))

    return np.array(option_list).reshape(col_idx.shape)

def cond_loss(data,output_info,cond_vec,m):

    tmp_loss = []
    
    start = 0
    start_cond = 0

    for item in output_info:
        if item[1] == 'tanh':
            start += item[0]
            continue
        elif item[1] == 'softmax':
            end = start + item[0]
            end_cond = start_cond + item[0]

            tmp = F.cross_entropy(data[:,start:end],
                                  torch.argmax(c[:,start_cond:end_cond],dim=1),
                                  reduction='none')
            tmp_loss.append(tmp)

            start = end
            start_cond = end_cond
    
    tmp_loss = torch.stack(tmp_loss,axis=1)
    loss = (tmp_loss * m).sum() / data.size()[0]

    return loss

def get_start_end(target_col_idx,ouptut_info):

    start = 0
    
    cate_c = 0
    all_c = 0

    for item in ouptut_info:
        if cate_c == target_col_idx:
            break
        if item[1] == 'tanh':
            start += item[0]
        elif item[1] == 'softmax':
            start += item[0]
            cate_c += 1
        all_c += 1
    
    end = start + ouptut_info[all_c][0]

    return (start,end)

def apply_activate(data,output_info):

    data_t = list()

    start = 0
    for item in output_info:
        if item[1] == 'tanh':
            end = start + item[0]
            data_t.append(torch.tanh(data[:,start:end]))
            start = end
        elif item[1] == 'softmax':
            end = start + item[0]
            # here use gumbel_softmax since argmax operation is not differentiable
            data_t.append(F.gumbel_softmax(data[:,start:end],tau=0.2))
            start = end
    
    act_data = torch.cat(data_t,dim=1)

    return act_data

def weights_init(model):
    classname = model.__class__.__name__

    if classname.find('Conv') != -1:
        init.normal_(model.weight.data,0.0,0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(model.weight.data,1.0,0.02)
        init.constant_(model.bias.data,0)

def build_discriminator_layers(side,num_channels):
    
    layer_dims = [(1,side),(num_channels,side//2)]

    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2,layer_dims[-1][1] // 2))
    
    layers_D = list()
    for prev,curr in zip(layer_dims,layer_dims[1:]):
        layers_D += [Conv2d(prev[0],curr[0],4,2,1,bias=False), # 卷积核=4,步长=2,padd=1 保证图像尺寸 * 1/2
                     BatchNorm2d(curr[0]),
                     LeakyReLU(0.2,inplace=True)]
    
    # last layer 输出一个numric value --- use sigmoid()
    layers_D += [Conv2d(layer_dims[-1][0],1,layer_dims[-1][1],1,0),Sigmoid()]

    return layers_D

def build_generator_layers(side,rand_dim,num_channels):

    layer_dims = [(1,side),(num_channels,side // 2)]

    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2,layer_dims[-1][1] // 2))
    
    layers_G = [ConvTranspose2d(rand_dim,layer_dims[-1][0],layer_dims[-1][1],1,0,output_padding=0,bias=False)]
    
    for prev,curr in zip(reversed(layer_dims),reversed(layer_dims[:-1])):
        layers_G += [BatchNorm2d(prev[0]),ReLU(True),ConvTranspose2d(prev[0],curr[0],4,2,1,output_padding=0,bias=True)]

    return layers_G   


class Cond_vector():

    def __init__(self,data,output_info):
        '''
        data:transformed data
        output_info:
        '''

        self.model = list()
        self.interval = list()
        self.n_col = 0 # num of one-hot-encoding representations
        self.n_opt = 0 # num of distinct categories across all one-hot-encoding representations
        self.p_log_sampling = list()
        self.p_sampling = list()

        start = 0
        for item in output_info:
            if item[1] == 'tanh':
                start += item[0]
                continue
            elif item[1] == 'softmax':
                end = start + item[0]
                self.model.append(np.argmax(data[:,start:end],axis=-1))
                self.interval.append((self.n_opt,item[0]))

                self.n_col += 1
                self.n_opt += item[0]
                
                freq = np.sum(data[:,start:end],axis=0)
                log_freq = np.log(freq + 1)
                log_pmf = log_freq / np.sum(log_freq)
                self.p_log_sampling.append(log_pmf)
                
                pmf = freq / np.sum(freq)
                self.p_sampling.append(pmf)

                start = end
        
        self.interval = np.asarray(self.interval)

    def train_sample(self,batch):

        if self.n_col == 0:
            return None
        batch = batch

        cond_vec = np.zeros((batch,self.n_opt),dtype = 'float32')

        idx = np.random.choice(np.arange(self.n_col),batch)

        mask = np.zeros((batch,self.n_col),dtype='float32')
        mask[np.arange(batch),idx] = 1

        category_val = col_based_sampling(self.p_log_sampling,idx)

        for i in np.arange(batch):
            cond_vec[i,self.interval[idx[i],0] + category_val[i]] = 1
        
        return cond_vec,mask,idx,category_val
    
    def sample(self,batch):

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
    # 抽样特定condition vector的样本
    
    def __init__(self,data,output_info):
        super(Sampler,self).__init__()

        self.data = data
        self.model = []
        self.n =len(data)

        start = 0
        for item in output_info:
            if item[1] == 'tanh':
                start += item[0]
                continue
            elif item[1] == 'softmax':
                end = start + item[0]
                tmp = []
                for j in range(item[0]):
                    tmp.append(np.nonzero(data[:,start + j])[0])

                self.model.append(tmp)
                start = end
    
    def sample(self,n,col,opt):

        if not col:
            idx = np.random.choice(np.arange(self.n),n)
            return self.data[idx]
        
        idx = []
        for c,v in zip(col,opt):
            idx.append(np.random.choice(self.model[c][v]))
        
        return self.data[idx]

class Classifier(nn.Module):

    def __init__(self,input_dim,class_dims,start_end):
        super(Classifier,self).__init__()
        self.input_dim = input_dim - (start_end[1] - start_end[0])
        self.class_dims = class_dims # hidden layer dims
        self.start_end = start_end

        layer_seq = list()
        hid_dim = self.input_dim

        for i in list(self.class_dims):
            layer_seq += [Linear(hid_dim,i),LeakyReLU(0.3),Dropout(0.5)]
            hid_dim = i
        
        if (start_end[1] - start_end[0]) == 2: # 2分类
            layer_seq += [Linear(hid_dim,1),Sigmoid()]
        else: # 多分类
            layer_seq += [Linear(hid_dim,(start_end[1]-start_end[0]))]
        
        self.seq = Sequential(*layer_seq)
    
    def forward(self,x):
        
        label = torch.argmax(x[:,self.start_end[0]:self.start_end[1]],axis=1)

        x_ = torch.cat((x[:,:self.start_end[0]],x[:,self.start_end[1]:]),1)

        if ((self.start_end[1] - self.start_end[0])==2):
            return self.seq(x_).view(-1),label
        else:
            return self.seq(x_),label


class Discriminator(nn.Module):

    def __init__(self,layers):
        super(Discriminator,self).__init__()
        self.seq = Sequential(*layers)
        self.seq_info = Sequential(*layers[:len(layers)-2])
    
    def forward(self,x):
        return (self.seq(x)),self.seq_info(x)

class Generator(nn.Module):

    def __init__(self,layers):
        super(Generator,self).__init__()
        self.seq = Sequential(*layers)
    
    def forward(self,x):
        return self.seq(x)

class Faker():

    def __init__(self,
                 rand_dim = 100,
                 class_dim = (256,256,256,256),
                 num_channels = 64,
                 l2scale = 1e-5,
                 batch_size = 512,
                 epochs=1):
        
        self.rand_dim = rand_dim
        self.class_dim = class_dim
        self.num_channels = num_channels
        self.d_side = None
        self.g_side = None
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.generator = None
        self.discriminator = None
        self.classifier = None
    
    def fit(self,train_data = pd.DataFrame,categorial=[],mixed={},ml_task={}):

        problem_type = None
        target_index = None

        if ml_task:
            problem_type = list(ml_task.keys())[0]
            if problem_type:
                target_index = train_data.columns.get_loc(ml_task[problem_type])
        
        self.transformer = D_Transformer(train_data=train_data,categorical_cols=categorial,mixed_dict=mixed)
        self.transformer.fit()

        # 编码转换
        train_data = self.transformer.transform(train_data.values)
        data_dim = self.transformer.output_dim

        # init
        data_sampler = Sampler(train_data,self.transformer.output_info)
        self.cond_generator = Cond_vector(train_data,self.transformer.output_info)

        # for discriminator
        sides_d = [4,8,16,24,32]
        col_size_d = data_dim + self.cond_generator.n_opt
        for i in sides_d:
            if i**2 >= col_size_d:
                self.dside = i
                break
        
        # for generator
        sides_g = [4,8,16,24,32]
        col_size_g = data_dim
        for i in sides:
            if i**2 >= col_size_g:
                self.g_side = i
                break
        
        layers_G = build_generator_layers(self.g_side,self.rand_dim+self.cond_generator.n_opt,self.num_channels)
        layers_D = build_discriminator_layers(self.d_side,self.num_channels)

        self.generator = Generator(layers_G).to(self.device)
        self.discriminator = Discriminator(layers_D).to(self.device)

        opt_params = dict(lr=1e-5, betas=(.5,.9), eps=1e-3, weight_decay=self.l2scale)
        opt_G = Adam(self.generator.parameters(),**opt_params)
        opt_D = Adam(self.discriminator.parameters(),**opt_params)


        # target col
        start_end = None
        classifier = None
        opt_C = None

        if target_index:
            start_end = get_start_end(target_index,self.transformer.output_info)
            
            self.classifier = Classifier(data_dim,self.class_dim,start_end).to(self.device)
            opt_C = Adam(self.classifier.parameters(),**opt_params)
        
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        # init img transformer
        self.G_transformer = Img_Transformer(self.g_side)
        self.D_transformer = Img_Transformer(self.d_side)


        steps_per_epoch = max(1,len(train_data)) // self.batch_size
        for i in tqdm(range(self.epochs)):
            for _ in range(steps_per_epoch):

                noisez = torch.randn(self.batch_size,self.rand_dim,device=self.device)
                cond_vec = self.cond_generator.train_sample(self.batch_size)

                c,m,col,opt = cond_vec
                c = torch.from_numpy(c).to(self.device)
                m = torch.from_numpy(m).to(self.device)

                noisez = torch.cat([noisez,c], dim=1) # 合并 noise & cond_vec
                noisez = noisez.view(self.batch_size,self.rand_dim+self.cond_generator.n_opt,1,1)

                real_idx = np.arange(self.batch_size)
                np.random.shuffle(real_idx)
                real = data_sampler.sample(self.batch_size,col[real_idx],opt[real_idx])
                real = real.from_numpy(real.astype('float32')).to(self.device)

                c_real = c[real_idx]

                # use generator to generate fake images
                fake = self.generator(noisez)
                # transform it into table 
                fake_t = self.G_transformer.inverse_transform(fake)
                # apply activation
                act_fake_t = apply_activate(fake_t,self.transformer.output_info)
                
                # concate with cond vec
                fake_cat = torch.cat([act_fake_t,c],dim=1)
                real_cat = torch.cat([real,c_real],dim=1)

                fake_cat_d = self.D_transformer.transform(fake_cat)
                real_cat_d = self.D_transformer.transform(real_cat)

                ################################# update Discriminator ##################################

                # delete cum gradient before each gradient descent
                opt_D.zero_grad()
                # apply discriminator to real input & fake input
                y_real,_ = self.discriminator(real_cat_d)
                y_fake,_ = self.discriminator(fake_cat_d)

                # compute loss for discriminator of GAN --- max likelihood log(D(x)) + log(1 - D(G(z))) 
                # ---> min loss_d = -log(D(x)) - log(1-D(G(z)))
                loss_d = (-(torch.log(y_real + 1e-4).mean()) - (torch.log(1. - y_fake + 1e-4).mean()))

                loss_d.backward() # bp
                opt_D.step() # gradient update

                ##################################### Done #############################################

                noisez = torch.randn(self.batch_size,self.rand_dim,device=self.device)
                cond_vec = self.cond_generator.train_sample(self.batch_size)

                c,m,col,opt = cond_vec
                c = torch.from_numpy(c).to(self.device)
                m = torch.from_numpy(m).to(self.device)

                noisez = torch.cat([noisez,c], dim=1) # 合并 noise & cond_vec
                noisez = noisez.view(self.batch_size,self.rand_dim+self.cond_generator.n_opt,1,1)

                ################################### update Generator ###################################

                opt_G.zero_grad()

                fake = self.generator(noisez)
                fake_t = self.G_transformer.inverse_transform(fake)
                act_fake_t = apply_activate(fake_t,self.transformer.output_info)
                fake_cat = torch.cat([act_fake_t,c],dim=1)
                fake_cat_d = self.D_transformer.transform(fake_cat)

                y_fake,info_fake = self.discriminator(fake_cat_d)
                _,info_real = self.discriminator(real_cat_d)
                conditional_loss = cond_loss(fake_t,self.transformer.output_info,c,m)

                # generator loss part1 = condition loss + - log(D(G(z)))
                loss_g = conditional_loss - (torch.log(y_fake + 1e-5).mean())
                loss_g.backward(retain_graph=True) # keep computation graph since we also have the information loss for gradient descent

                # generator loss part2 = information loss 
                # here we check the first-order and second-order statistics of fake data & real data
                loss_mean = torch.norm(torch.mean(info_fake.view(self.batch_size,-1),dim=0) - torch.mean(info_real.view(self.batch_size,-1),dim=0),1)
                loss_std = torch.norm(torch.std(info_fake.view(self.batch_size,-1),dim=0) - torch.std(info_real.view(self.batch_size,-1),dim=0),1)

                loss_info = loss_mean + loss_std

                loss_info.backward()

                opt_G.step()


                # check the target column for classification loss
                if problem_type:

                    c_loss = None

                    if (start_end[1] - start_end[0]) == 2: # binary classification problem
                        c_loss = BCELoss()
                    else: # multi-class 
                        c_loss = CrossEntropyLoss() 
                    
                    opt_C.zero_grad()
                    real_pred,real_label = self.classifier(real)
                    if (start_end[1] - start_end[0]) == 2:
                        real_label = real_label.type_as(real_pred)
                    
                    loss_classification = c_loss(real_pred,real_label)
                    loss_classification.backward()
                    opt_C.step()


                    # also update the weight of generator
                    opt_G.zero_grad()
                    fake = self.generator(noisez)
                    fake_t = self.G_transformer.inverse_transform(fake)
                    act_fake_t = apply_activate(fake_t,self.transformer.output_info)

                    fake_pred,fake_label = self.classifier(act_fake_t)
                    if (start_end[1] - start_end[0]) == 2:
                        fake_label = fake_label.type_as(fake_pred)
                    
                    loss_classification_g = c_loss(fake_pred,fake_label)
                    loss_classification_g.backward()
                    opt_G.step()

    def sample(self,num):
        
        # turn the generator into inference mode
        self.generator.eval()
        # col info
        output_info = self.transformer.output_info

        steps = n // self.batch_size + 1
        data = []

        for _ in range(steps):
            noisez = torch.randn(self.batch_size,self.rand_dim,device=self.device)
            cond_vec = self.cond_generator.sample(self.batch_size)
            c = cond_vec
            c = torch.from_numpy(c).to(self.device)
            noisez = torch.cat([noisez,c],dim=1)
            
            fake = self.generator(noisez)
            fake_t = self.G_transformer.inverse_transform(fake)
            act_fake_t = apply_activate(fake_t,output_info)

            data.append(act_fake_t.detach().cpu.numpy())
        
        data = np.concatenate(data,axis=0)
        result = self.transformer.inverse_transform(data)

        return result[0:num]

    























                








































    
        

        




        



    













    




    




















