import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.Models import get_non_pad_mask
import pdb
import numpy as np

@torch.jit.script
def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))

@torch.jit.script
def compute_event(event):
    """ Log-likelihood of events. """
    # add 1e-9 in case some events have 0 likelihood
    event += math.pow(10, -9)
    result = torch.log(event)
    return result

def compute_integral_unbiased(model, data, input, target):#, gap_f):
    """ Log-likelihood of non-events, using Monte Carlo integration. 
        data(B,1,M)
        input[B,seq-1]
        target[B,1]
        enc_out (B,seq-1,M)
    """
    
    #THP用
    #tauのrandom値を渡して、encoder2にtempencしてやる必要があるのでは？x
    num_samples = 500
    #random samples 
    rand_time = target.unsqueeze(2) * \
                torch.rand([*target.size(), num_samples], device=data.device)#[B,1,num_samples]
    
    #rand_time /= (target + 1)#[B,M]
    #temp_output = model.decoder(rand_time,enc_out,enc_out,None)
    #B,100,M
    
    temp_hid = model.linear(data[:,-2:-1,:])#[B,1,1]
    temp_lambda = softplus(temp_hid + model.alpha * rand_time,model.beta)#[B,1,samples]
    all_lambda = torch.sum(temp_lambda,dim=2)/num_samples#[B,1]
    unbiased_integral = all_lambda * target #[B,1]
    
    
    return unbiased_integral

def log_likelihood(model, output, input, target):#, gap_f):
    """ Log-likelihood of sequence. """

    #B*1*M output
    all_hid = model.linear(output[:,-1:,:])
    #B*1*1
    all_lambda = softplus(all_hid,model.beta)
    all_lambda = torch.sum(all_lambda,dim=2)#(B,sequence,type)の名残
    #[B*1]

    # event log-likelihood
    event_ll = compute_event(all_lambda)#[B,1]
    event_ll = torch.sum(event_ll,dim=-1)#[B]
    #B*1*1
    # non-event log-likelihood, either numerical integration or MC integration
    # non_event_ll = compute_integral_biased(, time, non_pad_mask)
    non_event_ll = compute_integral_unbiased(model, output, input, target)#[16,1]
    non_event_ll = torch.sum(non_event_ll, dim=-1)#[B]
    return event_ll, non_event_ll

def time_loss_se(prediction, input, target):
    #prediction : (B,L-1,1)
    #event_time: (B,L)
    prediction = prediction[:,-1].squeeze(-1)
    target = target.reshape(prediction.shape)
    #t_1~t_L
    """ Time prediction loss. """
    # event time gap prediction
    diff = prediction - target
    se = torch.sum(diff * diff)
    return se

def time_loss_ae(prediction, input, target):
    prediction = prediction[:,-1].squeeze(-1)  
    target = target.reshape(prediction.shape)  
    # event time gap prediction
    diff = prediction - target
    ae = torch.sum(torch.abs(diff))
    return ae

def time_mean_prediction(model, output, input, target, opt):
    #output[B,1,M], input[B,seq]
    left= math.pow(10,-9)*torch.ones(target.shape,dtype=torch.float64,device=opt.device)
    #[B,1]
    right=opt.train_mean*100*torch.ones(target.shape,dtype=torch.float64,device=opt.device)
    #[B,1]
    #input = torch.cat((input,target),dim=1)#THP用
    for _ in range(0,23):
        
        #THP用
        center=(left+right)/2
        center = center.reshape(target.shape)
        output, _ = model(input,center)
        _,non_event_ll=log_likelihood(model, output, input, center)
        value= non_event_ll-np.log(2)
        value = value.reshape(target.shape)#B,1
        left = (torch.where(value<0,center,left))#.unsqueeze(1)
        right = (torch.where(value>=0, center, right))#.unsqueeze(1)
    return (left+right)/2