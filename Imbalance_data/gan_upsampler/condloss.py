import torch
import torch.nn.functional as F

def cond_loss(data,output_info,cond_vec,m):
    '''
    计算conditional loss

    data:重编码转换后的表格数据
    output_info:对原始特征重编码后的相关信息 (feature value & mode type)
    cond_vec: 条件向量 [0,0,0,1,0,0,...]
    m: 记录数据中每条样本对应的独热编码1来自哪个col
    '''

    tmp_loss = []
    
    start = 0 # 数据行 起始位置指针
    start_cond = 0 # cond vec 起始位置指针

    for item in output_info:
        if item[1] == 'tanh':
            start += item[0]
            continue
        elif item[1] == 'softmax':
            end = start + item[0]
            end_cond = start_cond + item[0]

            # 计算局部loss
            tmp = F.cross_entropy(data[:,start:end],
                                  torch.argmax(cond_vec[:,start_cond:end_cond],dim=1),
                                  reduction='none')
            tmp_loss.append(tmp)

            start = end
            start_cond = end_cond
    
    tmp_loss = torch.stack(tmp_loss,axis=1) # 合并每个col对应的loss
    loss = (tmp_loss * m).sum() / data.size()[0]

    return loss