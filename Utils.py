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
def THP_compute_integral_unbiased(model, data, input, target):#, gap_f):
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
    temp_output = data[:,-2:-1,:]
    for linear_layer in model.layer_stack:
        temp_output = linear_layer(temp_output)
    temp_hid = temp_output#[B,1,1]
    temp_lambda = softplus(temp_hid + model.alpha * rand_time,model.beta)#[B,1,samples]
    all_lambda = torch.sum(temp_lambda,dim=2)/num_samples#[B,1]
    unbiased_integral = all_lambda * target #[B,1]
    
    
    return unbiased_integral

def compute_integral_unbiased(model, data, input, target, enc_out):#, gap_f):
    """ Log-likelihood of non-events, using Monte Carlo integration. 
        data(B,1,M)
        input[B,seq-1]
        target[B,1]
        enc_out (B,1,M)
    """
    
    #if model.anchor_vector.requires_grad==True:
    #    torch.manual_seed(42)
    num_samples = 500
    #random samples 
    rand_time = target * torch.rand([target.shape[0],num_samples], device=data.device)
    #B,500
    #temp_output = model.decoder(rand_time,enc_out,enc_out,None)
    #pdb.set_trace()
    temp_output = model.decoder(rand_time,k=enc_out,v=enc_out)
    #B,500,M
    # temp_hid = model.linear2(temp_output)
    # relu_hid = model.relu(temp_hid)
    # a_lambda = model.linear(relu_hid)
    for linear_layer in model.layer_stack:
        temp_output = linear_layer(temp_output)
    a_lambda = temp_output
    #B,500,1
    temp_lambda = softplus(a_lambda,model.beta)
    #B,500,1
    all_lambda = torch.sum(temp_lambda,dim=1)/num_samples    
    #B,1
    unbiased_integral = all_lambda * target #/target
    return unbiased_integral

def log_likelihood(model, output, input, target, enc_out):#, gap_f):
    """ Log-likelihood of sequence. """
    if model.method=="THP":
        temp_output = output[:,-1:,:]
        for linear_layer in model.layer_stack:
            temp_output = linear_layer(temp_output)
        #B*1*M output
        all_hid = temp_output
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
        non_event_ll = THP_compute_integral_unbiased(model, output, input, target)#[16,1]
        non_event_ll = torch.sum(non_event_ll, dim=-1)#[B]
        return event_ll, non_event_ll
    
    #output 256,1,64
    #B*1*M output
    for linear_layer in model.layer_stack:
        output = linear_layer(output)
          
    # temp_hid = model.linear2(output)
    # relu_hid = model.relu(temp_hid)
    # all_hid = model.linear(relu_hid)
    all_hid = output
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
    non_event_ll = compute_integral_unbiased(model, output, input, target, enc_out)#[16,1]
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

def time_mean_prediction(model, output, input, target, enc_out, opt):
    #output[B,1,M], input[B,seq]
    left=opt.train_mean*0.0001*torch.ones(target.shape,dtype=torch.float64,device=opt.device)
    #[B,1]
    right=opt.train_mean*100*torch.ones(target.shape,dtype=torch.float64,device=opt.device)
    #[B,1]
    #input = torch.cat((input,target),dim=1)#THP用
    for _ in range(0,13):
        """
        #THP用
        center=(left+right)/2

        center = center.reshape(target.shape)
        output, _, enc_out = model(input,center)
        _, non_event_ll = log_likelihood(model, output, input, center, enc_out)
        value= non_event_ll-np.log(2)
        value = value.reshape(target.shape)#B,1
        left = (torch.where(value<0,center,left))#.unsqueeze(1)
        right = (torch.where(value>=0, center, right))#.unsqueeze(1)
        """
        
        
        #vec.pool用
        center=(left+right)/2
        output, _, enc_out = model(input,center)
        _, non_event_ll = log_likelihood(model, output, input, center, enc_out)
        value= non_event_ll-np.log(2)
        value = value.reshape(target.shape)#B,1
        left = (torch.where(value<0,center,left))
        right = (torch.where(value>=0, center, right))
        
    return (left+right)/2