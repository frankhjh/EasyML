import numpy as np
import torch
import torch.nn.functional as F

def col_based_sampling(probs,col_idx):

    option_list =[]

    for i in col_idx:
        p = probs[i] + 1e-5
        p = p/sum(p)

        option_list.append(np.random.choice(np.arange(len(probs[i])),p=p))

    return np.array(option_list).reshape(col_idx.shape)


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