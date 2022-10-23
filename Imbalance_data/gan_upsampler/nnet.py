import numpy as np
import torch
import torch.nn as nn
from torch.nn import Dropout
from torch.nn import LeakyReLU,ReLU
from torch.nn import init
from torch.nn import BatchNorm2d,BatchNorm1d
from torch.nn import Linear,Conv2d,ConvTranspose2d
from torch.nn import Sigmoid,Softmax
from torch.nn import Sequential


def weights_init(model):
    classname = model.__class__.__name__

    if classname.find('Conv') != -1:
        init.normal_(model.weight.data,0.0,0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(model.weight.data,1.0,0.02)
        init.constant_(model.bias.data,0)

class Residual(nn.Module):
    '''
    residual net
    '''
    def __init__(self,input_dim,output_dim):
        super(Residual,self).__init__()
        self.fc = Linear(input_dim,output_dim)
        self.bn = BatchNorm1d(output_dim)
        self.relu = ReLU()
    
    def forward(self,x):
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)

        return torch.cat([out,x],dim=1)



class Discriminator(nn.Module):
    '''
    判别器
    '''

    def __init__(self,side,num_channels):
        super(Discriminator,self).__init__()

        layers = self.build_layers(side,num_channels)

        self.seq = Sequential(*layers)
        self.seq_info = Sequential(*layers[:len(layers)-2])
    
    
    def build_layers(self,side,num_channels):
        layer_dims = [(1,side),(num_channels,side//2)]

        while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
            # 从第二层开始，每经过一层卷积，channel数 * 2，图片长宽 * 1/2
            layer_dims.append((layer_dims[-1][0] * 2,layer_dims[-1][1] // 2))
        
        layers_D = list()
        for prev,curr in zip(layer_dims,layer_dims[1:]):
            # 每一层卷积的 input_channel = prev[0], output_channel = curr[0], kernel_size = 4, stride = 2, padding =1
            layers_D += [Conv2d(prev[0],curr[0],4,2,1,bias=False), # 卷积核=4,步长=2,padd=1 保证图像尺寸 * 1/2
                        BatchNorm2d(curr[0]),
                        LeakyReLU(0.2,inplace=True)]
        
        # 最后一个卷积层输出一个channel,图像尺寸 1*1 --> 即一个numerical value, 再经过一个sigmoid激活函数得到最终输出。
        layers_D += [Conv2d(layer_dims[-1][0],1,layer_dims[-1][1],1,0),Sigmoid()]

        return layers_D

    def forward(self,x):
        return (self.seq(x)),self.seq_info(x)


class MLP_Discriminator(nn.Module):
    '''
    基于MLP的判别器尝试
    '''
    def __init__(self,input_dim,num_hid):
        super(MLP_Discriminator,self).__init__()

        layers = self.build_layers(input_dim,num_hid)

        self.mlp_d_seq = Sequential(*layers)
        self.mlp_d_seq_info = Sequential(*layers[:len(layers)-2])

    
    def build_layers(self,input_dim,num_hid):
        
        hidden_dims = [input_dim]
        layers_D = list()
       
        while hidden_dims[-1] > 1 and len(hidden_dims) < num_hid + 1:
            hidden_dims.append(hidden_dims[-1]//2)
        
        for prev,curr in zip(hidden_dims,hidden_dims[1:]):
            layers_D += [Linear(prev,curr),LeakyReLU(0.2,inplace=True),Dropout(0.3)]
        
        layers_D += [Linear(hidden_dims[-1],1),Sigmoid()]

        return layers_D
    
    def forward(self,x):
        return self.mlp_d_seq(x),self.mlp_d_seq_info(x)


class Attn_MLP_Discriminator(nn.Module):
    '''
    在MLP中引入注意力机制
    '''
    def __init__(self,input_dim,num_hid):
        super(Attn_MLP_Discriminator,self).__init__()

        self.attn_layer = Linear(input_dim,input_dim)
        self.relu = ReLU()
        self.softmax = Softmax(dim=1)

        layers = self.build_layers(input_dim,num_hid)

        self.mlp_d_seq = Sequential(*layers)
        self.mlp_d_seq_info = Sequential(*layers[:len(layers)-2])

    
    def build_layers(self,input_dim,num_hid):
        
        hidden_dims = [input_dim]
        layers_D = list()
       
        while hidden_dims[-1] > 1 and len(hidden_dims) < num_hid + 1:
            hidden_dims.append(hidden_dims[-1]//2)
        
        for prev,curr in zip(hidden_dims,hidden_dims[1:]):
            layers_D += [Linear(prev,curr),LeakyReLU(0.2,inplace=True),Dropout(0.3)]
        
        layers_D += [Linear(hidden_dims[-1],1),Sigmoid()]

        return layers_D
    
    def forward(self,x):

        attn = self.attn_layer(x)
        attn = self.relu(attn)
        attn = self.softmax(attn)

        x = torch.mul(x,attn)

        return self.mlp_d_seq(x),self.mlp_d_seq_info(x)       

        
class Generator(nn.Module):
    '''
    生成器
    '''
    def __init__(self,side,rand_dim,num_channels):
        super(Generator,self).__init__()
        
        layers = self.build_layers(side,rand_dim,num_channels)

        self.seq = Sequential(*layers)
    
    def build_layers(self,side,rand_dim,num_channels):
        layer_dims = [(1,side),(num_channels,side // 2)]

        while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
            layer_dims.append((layer_dims[-1][0] * 2,layer_dims[-1][1] // 2))
        
        # 初始反卷积：输入渠道数 = rand_dim, 输出渠道数 = layer_dims 中最后位置对应的渠道数，kernel_size = layer_dims 中最后位置对应的卷积扩充核大小
        layers_G = [ConvTranspose2d(rand_dim,layer_dims[-1][0],layer_dims[-1][1],1,0,output_padding=0,bias=False)]
        
        for prev,curr in zip(reversed(layer_dims),reversed(layer_dims[:-1])):
            # 每经过一层反卷积，输出channel变为原来的1/2（最后一层除外),在最后一层，由num_channel--> 1。
            #  图像长宽不断 * 2 直到最终输出尺寸 = side
            layers_G += [BatchNorm2d(prev[0]),ReLU(True),ConvTranspose2d(prev[0],curr[0],4,2,1,output_padding=0,bias=True)]

        return layers_G 

    def forward(self,x):
        return self.seq(x)


class MLP_Generator(nn.Module):
    '''
    基于MLP的生成器尝试
    '''
    def __init__(self,rand_dim,out_dim,num_hid):
        super(MLP_Generator,self).__init__()

        layers = self.build_layers(rand_dim,out_dim,num_hid)

        self.mlp_g_seq = Sequential(*layers)
    
    def build_layers(self,rand_dim,out_dim,num_hid):
        '''
        rand_dim: generator输入维度 = len(noisez) + len(cond_vec)
        out_dim: generator输出维度
        num_hid:hidden layer数
        '''
        hidden_dims = [rand_dim]
        layers_G = list()

        delta = np.abs(out_dim - rand_dim) // num_hid
        for i in range(num_hid):
            hidden_dims.append(rand_dim + i * delta)
        
        hidden_dims += [out_dim]

        for prev,curr in zip(hidden_dims,hidden_dims[1:]):
            layers_G += [Linear(prev,curr),BatchNorm1d(curr),ReLU(True)]
        
        return layers_G
    
    def forward(self,x):
        return self.mlp_g_seq(x)


class Attn_MLP_Generator(nn.Module):
    '''
    在MLP中引入注意力机制
    '''
    def __init__(self,rand_dim,out_dim,num_hid):
        super(Attn_MLP_Generator,self).__init__()

        self.attn_layer = Linear(rand_dim,rand_dim)
        self.relu = ReLU()
        self.softmax = Softmax(dim=1)

        layers = self.build_layers(rand_dim,out_dim,num_hid)

        self.mlp_g_seq = Sequential(*layers)
    
    def build_layers(self,rand_dim,out_dim,num_hid):
        '''
        rand_dim: generator输入维度 = len(noisez) + len(cond_vec)
        out_dim: generator输出维度
        num_hid:hidden layer数
        '''
        hidden_dims = [rand_dim]
        layers_G = list()

        attn_layer = Linear(rand_dim,rand_dim)

        delta = np.abs(out_dim - rand_dim) // num_hid
        for i in range(num_hid):
            hidden_dims.append(rand_dim + i * delta)
        
        hidden_dims += [out_dim]

        for prev,curr in zip(hidden_dims,hidden_dims[1:]):
            layers_G += [Linear(prev,curr),BatchNorm1d(curr),ReLU(True)]
        
        return layers_G
    
    def forward(self,x):

        attn = self.attn_layer(x)
        attn = self.relu(attn)
        attn = self.softmax(attn)

        x = torch.mul(x,attn)
        return self.mlp_g_seq(x)

    

class Classifier(nn.Module):
    '''
    分类器:基于真实样本进行模型训练。 然后对generator生成的【虚假样本】判断生成的label是否符合逻辑。
    '''

    def __init__(self,input_dim,class_dims,start_end):
        super(Classifier,self).__init__()
        '''
        input_dim: 输入特征的维度
        class_dims: 隐藏层神经元数
        start_end: 标签的起始和结束位置
        '''
        self.input_dim = input_dim - (start_end[1] - start_end[0]) # 剔除标签对应的独热表征
        self.class_dims = class_dims 
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
        
        label = torch.argmax(x[:,self.start_end[0]:self.start_end[1]],axis=1) # 训练标签

        x_ = torch.cat((x[:,:self.start_end[0]],x[:,self.start_end[1]:]),1) # 剔除标签后合并所有特征

        if ((self.start_end[1] - self.start_end[0])==2):
            return self.seq(x_).view(-1),label  # 同时返回预测结果 & 标签
        else:
            return self.seq(x_),label 


