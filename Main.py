import os
import argparse
import numpy as np
import pandas as pd
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
import transformer.Constants as Constants
import Utils
import data.datagen

from matplotlib import pyplot as plt
import plot_code
from transformer.Models import Transformer
from tqdm import tqdm

from scipy.stats import lognorm,gamma
from scipy.optimize import brentq
from datetime import datetime as dt

######################################################
### Data Generator
######################################################
def generate_stationary_poisson():
    np.random.seed(seed=32)
    tau = np.random.exponential(size=100000)
    T = tau.cumsum()
    score = 1
    return [T,score]

def generate_nonstationary_poisson():
    np.random.seed(seed=32)
    L = 20000
    amp = 0.99
    l_t = lambda t: np.sin(2*np.pi*t/L)*amp + 1
    l_int = lambda t1,t2: - L/(2*np.pi)*( np.cos(2*np.pi*t2/L) - np.cos(2*np.pi*t1/L) )*amp   + (t2-t1)
    
    while 1:
        T = np.random.exponential(size=210000).cumsum()*0.5
        r = np.random.rand(210000)
        index = r < l_t(T)/2.0
        
        if index.sum() > 100000:
            T = T[index][:100000]
            score = - ( np.log(l_t(T[80000:])).sum() - l_int(T[80000-1],T[-1]) )/20000
            break
       
    return [T,score]

def generate_stationary_renewal():
    np.random.seed(seed=32)
    s = np.sqrt(np.log(6*6+1))
    mu = -s*s/2
    tau = lognorm.rvs(s=s,scale=np.exp(mu),size=100000)
    lpdf = lognorm.logpdf(tau,s=s,scale=np.exp(mu))
    T = tau.cumsum()
    score = - np.mean(lpdf[80000:])
    
    return [T,score]

def generate_nonstationary_renewal():
    np.random.seed(seed=32)
    L = 20000
    amp = 0.99
    l_t = lambda t: np.sin(2*np.pi*t/L)*amp + 1
    l_int = lambda t1,t2: - L/(2*np.pi)*( np.cos(2*np.pi*t2/L) - np.cos(2*np.pi*t1/L) )*amp   + (t2-t1)

    T = []
    lpdf = []
    x = 0

    k = 4
    rs = gamma.rvs(k,size=100000)
    lpdfs = gamma.logpdf(rs,k)
    rs = rs/k
    lpdfs = lpdfs + np.log(k)

    for i in range(100000):
        x_next = brentq(lambda t: l_int(x,t) - rs[i],x,x+1000)
        l = l_t(x_next)
        T.append(x_next)
        lpdf.append(  lpdfs[i] + np.log(l) )  
        x = x_next

    T = np.array(T)
    lpdf = np.array(lpdf)
    score = - lpdf[80000:].mean()
    
    return [T,score]

def generate_self_correcting():
    np.random.seed(seed=32)
    
    def self_correcting_process(mu,alpha,n):
    
        t = 0; x = 0;
        T = [];
        log_l = [];
        Int_l = [];
    
        for i in range(n):
            e = np.random.exponential()
            tau = np.log( e*mu/np.exp(x) + 1 )/mu # e = ( np.exp(mu*tau)- 1 )*np.exp(x) /mu
            t = t+tau
            T.append(t)
            x = x + mu*tau
            log_l.append(x)
            Int_l.append(e)
            x = x -alpha

        return [np.array(T),np.array(log_l),np.array(Int_l)]
    
    [T,log_l,Int_l] = self_correcting_process(1,1,100000)
    score = - ( log_l[80000:] - Int_l[80000:] ).sum() / 20000
    
    return [T,score]

def generate_hawkes1():
    np.random.seed(seed=32)
    [T,LL] = simulate_hawkes(100000,0.2,[0.8,0.0],[1.0,20.0])
    score = - LL[80000:].mean()
    return [T,score]

def generate_hawkes2():
    np.random.seed(seed=32)
    [T,LL] = simulate_hawkes(100000,0.2,[0.4,0.4],[1.0,20.0])
    score = - LL[80000:].mean()
    return [T,score]

def simulate_hawkes(n,mu,alpha,beta):
    T = []
    LL = []
    
    x = 0
    l_trg1 = 0
    l_trg2 = 0
    l_trg_Int1 = 0
    l_trg_Int2 = 0
    mu_Int = 0
    count = 0
    
    while 1:
        l = mu + l_trg1 + l_trg2
        step = np.random.exponential()/l
        x = x + step
        
        l_trg_Int1 += l_trg1 * ( 1 - np.exp(-beta[0]*step) ) / beta[0]
        l_trg_Int2 += l_trg2 * ( 1 - np.exp(-beta[1]*step) ) / beta[1]
        mu_Int += mu * step
        l_trg1 *= np.exp(-beta[0]*step)
        l_trg2 *= np.exp(-beta[1]*step)
        l_next = mu + l_trg1 + l_trg2
        
        if np.random.rand() < l_next/l: #accept
            T.append(x)
            LL.append( np.log(l_next) - l_trg_Int1 - l_trg_Int2 - mu_Int )
            l_trg1 += alpha[0]*beta[0]
            l_trg2 += alpha[1]*beta[1]
            l_trg_Int1 = 0
            l_trg_Int2 = 0
            mu_Int = 0
            count += 1
            
            if count == n:
                break
        
    return [np.array(T),np.array(LL)]

def generate_data(data_type,opt):
    def rolling_matrix(x,time_step):
            x = x.flatten()
            n = x.shape[0]
            stride = x.strides[0]
            return np.lib.stride_tricks.as_strided(x, shape=(n-time_step+1, time_step), strides=(stride,stride) ).copy()
    def transform_data(T,n_train,n_validation,n_test,time_step,batch_size,opt):

        T_train = T[:n_train]
        T_valid = T[n_train:n_train+n_validation]
        T_test = T[n_train+n_validation:n_train+n_validation+n_test]
        
        dT_train = np.ediff1d(T_train)
        opt.train_mean=dT_train.mean()
        train_data = torch.tensor(rolling_matrix(dT_train,time_step)).to(torch.double)
        
        dT_valid = np.ediff1d(T_valid)
        valid_data = torch.tensor(rolling_matrix(dT_valid,time_step)).to(torch.double)
        dT_test = np.ediff1d(T_test)
        test_data = torch.tensor(rolling_matrix(dT_test,time_step)).to(torch.double)
        return torch.utils.data.DataLoader(train_data,num_workers=os.cpu_count(),batch_size=batch_size,pin_memory=True,shuffle=True),\
        torch.utils.data.DataLoader(valid_data,num_workers=os.cpu_count(),batch_size=batch_size,pin_memory=True,shuffle=False), \
        torch.utils.data.DataLoader(test_data,num_workers=os.cpu_count(),batch_size=batch_size,pin_memory=True,shuffle=False)\
        ,dT_train.max(),dT_train.min(),np.median(dT_train)
    
    if data_type == 'sp':
        [T,score_ref] = generate_stationary_poisson()
    elif data_type == 'nsp':
        [T,score_ref] = generate_nonstationary_poisson()
    elif data_type == 'sr':
        [T,score_ref] = generate_stationary_renewal()
    elif data_type == 'nsr':
        [T,score_ref]=generate_nonstationary_renewal()
    elif data_type == 'sc':
        [T,score_ref]=generate_self_correcting()
    elif data_type == 'h1':
        [T,score_ref]=generate_hawkes1()
    elif data_type == 'h2':
        [T,score_ref]=generate_hawkes2()
    
    n = T.shape[0]
    time_step=opt.time_step
    batch_size=opt.batch_size

    trainloader, validloader, testloader ,train_max, train_min,train_med= transform_data(T,int(n*0.8),int(n*0.1),int(n*0.1),time_step,batch_size,opt) # A sequence is divided into training and test data.
    
    return trainloader, validloader,testloader,train_max,train_min,train_med
def set_date_class(df,opt):
    def rolling_matrix(x,time_step):
            x = x.flatten()
            n = x.shape[0]
            stride = x.strides[0]
            return np.lib.stride_tricks.as_strided(x, shape=(n-time_step+1, time_step), strides=(stride,stride) ).copy()
    
    
    df["dt64"] = df["dtt64"].map(pd.Timestamp.timestamp)/3600##UNIX変換
    df_train=df[:int(len(df)*0.8)]
    dT_train=np.ediff1d(df_train["dt64"])
    train_data = torch.tensor(rolling_matrix(dT_train,opt.time_step)).to(torch.double)
    train_dataset = torch.utils.data.TensorDataset(train_data)
    df_valid=df[int(len(df)*0.8):int(len(df)*0.9)]
    dT_valid=np.ediff1d(df_valid["dt64"])
    rT_valid = torch.tensor(rolling_matrix(dT_valid,opt.time_step)).to(torch.double)
    df_test = df[int(len(df)*0.9):]
    df_test = df_test.reset_index()
    
    dT_test = np.ediff1d(df_test["dt64"])
    rT_test = torch.tensor(rolling_matrix(dT_test,opt.time_step)).to(torch.double)
    
    trainloader = torch.utils.data.DataLoader(train_data,num_workers=os.cpu_count(),batch_size=opt.batch_size,pin_memory=True,shuffle=True)
    validloader = torch.utils.data.DataLoader(rT_valid,num_workers=os.cpu_count(),batch_size=opt.batch_size,pin_memory=True,shuffle=False)
    testloader = torch.utils.data.DataLoader(rT_test,num_workers=os.cpu_count(),batch_size=opt.batch_size,pin_memory=True,shuffle=False)
 
    opt.train_mean=dT_train.mean()
    train_max=dT_train.max()
    train_min=dT_train.min()
    train_med=np.median(dT_train)

    return trainloader, validloader, testloader,train_max,train_min,train_med

################
### Early Stop
################
class EarlyStopping:
    def __init__(self,patience=10, verbose=False, path='c_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.checkpoint(val_loss, model)
        elif score < self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1
            if self.verbose:  #表示を有b効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する 
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.checkpoint(val_loss, model)
            self.counter = 0
    def checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        self.val_loss_min = val_loss  #その時のlossを記録する

################
### Train
################
def train_epoch(model, training_data, optimizer, opt):
    """ Epoch operation in training phase. """
    model.train()
    
    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_time_ae = 0
    total_num_event = 0  # number of total events
    
    for batch in tqdm(training_data, mininterval=2,
                      desc='-(Training)  ', leave=False):
        
        """ prepare data """
        event_time = batch.to(opt.device, non_blocking=True)
        
        train_input = event_time[:,:-1]#[B,Seqence-1]
        train_target = event_time[:,-1:]#[B,1]
        
        """ forward """
        optimizer.zero_grad()
        #model_output [B,1,M], prediction[B,1], enc_out[B,Sequence,M]
        model_output, prediction, enc_out = model(train_input,train_target)
        
        
        """ backward """
        # negative log-likelihood
        #event_ll[B,1,1], non_event_ll.shape[B,1,1]
        
        event_ll, non_event_ll = Utils.log_likelihood(model, model_output, train_input, train_target, enc_out)
        event_loss = -torch.sum(event_ll - non_event_ll)#[]
        
        # time prediction
        with torch.no_grad():
            se = Utils.time_loss_se(prediction, train_input, train_target)#[]
        ae = Utils.time_loss_ae(prediction, train_input, train_target)#[]
        # SE is usually large, scale it to stabilize training
        scale_time_loss = opt.loss_scale
        
        loss = event_loss + ae / scale_time_loss 
        loss.backward()
        if opt.grad_log==True:
            for name, param in model.named_parameters():
                with open("./param_grad/"+opt.imp+"/"+name+".log","a") as f:
                    f.write('{grad}\n'.format(grad=param.grad))
        """ update parameters """
        optimizer.step()
        """ note keeping """
        total_event_ll += -event_loss.item()
        total_time_se += se.item()
        total_time_ae += ae.item()
        total_num_event += event_time.shape[0] 
        
    
    rmse = np.sqrt(total_time_se / total_num_event)
    mae = total_time_ae / total_num_event
    return total_event_ll / total_num_event, mae, rmse
################
### Evaluation
################
def eval_epoch(model, validation_data, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_time_ae = 0
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            event_time = batch.to(opt.device, non_blocking=True)
            #event_time = event_time.to(torch.float)
            train_input = event_time[:,:-1]
            train_target = event_time[:,-1:]
            
            """ forward """
            output, prediction, enc_out = model(train_input,train_target)
            """ compute loss """
            event_ll, non_event_ll = Utils.log_likelihood(model, output, train_input, train_target, enc_out)
            event_loss = -torch.sum(event_ll - non_event_ll)            
            
            # time prediction
            se = Utils.time_loss_se(prediction, train_input, train_target)
            ae = Utils.time_loss_ae(prediction, train_input, train_target)
            """ note keeping """
            total_event_ll += -event_loss.item()
            total_time_se += se.item()
            total_time_ae += ae.item()
            total_num_event += event_time.shape[0]
    rmse = np.sqrt(total_time_se / total_num_event)
    mae = total_time_ae / total_num_event
    return total_event_ll / total_num_event, mae, rmse

################
### train-eval-plot-earlystop
################
def train(model, training_data, validation_data ,test_data,optimizer, scheduler, opt):
    """ Start training. """
    train_loss_his = []

    valid_event_losses = []  # validation log-likelihood
    valid_mae_history = [] # validation event time prediction MAE
    valid_rmse_history = []  # validation event time prediction RMSE
    valid_loss_his = []
    
    if opt.train==True:
        torch.backends.cudnn.benchmark = True
        es = EarlyStopping(verbose=True,path="checkpoint/tau/"+opt.wp+'.pth')
    
        for epoch_i in range(opt.epoch):
            epoch = epoch_i + 1
            print('[ Epoch', epoch, ']')
            start = time.time()
            
            train_event, train_mae ,train_rmse= train_epoch(model, training_data, optimizer, opt)
            print('  - (Training)    Loss:{loss: 8.5f},loglikelihood: {ll: 8.5f}, '
                ' MAE: {mae: 8.5f},'
                'RMSE: {rmse: 8.5f}, '
                'elapse: {elapse:3.3f} min'
                .format(loss=-train_event+train_mae/opt.loss_scale, ll=train_event, mae=train_mae, rmse=train_rmse, elapse=(time.time() - start) / 60))
            train_loss_his += [-train_event+train_mae/opt.loss_scale]
            
            start = time.time()
            valid_event, valid_mae, valid_rmse = eval_epoch(model, validation_data, opt)
            print('  - (Valid   )    Loss:{loss: 8.5f},loglikelihood: {ll: 8.5f}, '
                ' MAE: {mae: 8.5f},'
                ' RMSE: {rmse: 8.5f}, '
                'elapse: {elapse:3.3f} min'
                .format(loss=-valid_event+ valid_mae/opt.loss_scale, ll=valid_event, mae=valid_mae, rmse=valid_rmse, elapse=(time.time() - start) / 60))
            valid_loss_his +=[-valid_event+ valid_mae/opt.loss_scale]
            valid_event_losses += [valid_event]
            valid_mae_history += [valid_mae]
            valid_rmse_history += [valid_rmse]
            print('  - [Info] Loss: {loss:8.5f}, Maximum ll: {event: 8.5f}, Minimum MAE: {mae: 8.5f}, Minimum RMSE:{rmse: 8.5f}'
                .format(loss=max(valid_loss_his), event=max(valid_event_losses), mae=min(valid_mae_history), rmse=min(valid_rmse_history)))

            # logging
            with open(opt.log, 'a') as f:
                f.write("train     : "+'{epoch}, {loss: 8.5f}, {ll: 8.5f}, {mae: 8.5f}, {rmse: 8.5f}\n'
                    .format(epoch=epoch,loss=-train_event+train_mae/opt.loss_scale, ll=train_event, mae=train_mae, rmse=train_rmse))
                f.write("validation: "+'{epoch}, {loss: 8.5f}, {ll: 8.5f}, {mae: 8.5f}, {rmse: 8.5f}\n\n'
                    .format(epoch=epoch,loss=-valid_event+ valid_mae/opt.loss_scale, ll=valid_event, mae=valid_mae, rmse=valid_rmse))

            scheduler.step()
            
            es( -valid_event+ valid_mae/opt.loss_scale ,model)
            if es.early_stop: #ストップフラグがTrueの場合、breakでforループを抜ける
                print("Early Stopping!")
                break

        model_path="checkpoint/tau/"+opt.wp+'.pth'
        model.load_state_dict(torch.load(model_path))
        model.eval()
        test_event, test_mae, test_rmse = eval_epoch(model, test_data, opt)
        print('  - (testing   )    Loss:{loss: 8.5f},loglikelihood: {ll: 8.5f}, '
                ' MAE: {mae: 8.5f},'
                ' RMSE: {rmse: 8.5f}, '
                'elapse: {elapse:3.3f} min'
                .format(loss=-test_event+ test_mae/opt.loss_scale, ll=test_event, mae=test_mae, rmse=test_rmse, elapse=(time.time() - start) / 60))
            
        with open(opt.log, 'a') as f:
            f.write("testing: "+'{epoch}, {loss: 8.5f}, {ll: 8.5f}, {mae: 8.5f}, {rmse: 8.5f}\n'
                    .format(epoch=epoch,loss=-test_event+ test_mae/opt.loss_scale, ll=test_event, mae=test_mae, rmse=test_rmse))
        
        with open(opt.log, 'a') as f:
            f.write('rep values {anchor}\n'.format(anchor=model.rep_vector))
            f.write('anchor values {anchor}\n'.format(anchor=model.anchor_vector))
        plot_code.plot_learning_curve(train_loss_his,valid_loss_his,opt)
    
    else:
        model_path="checkpoint/tau/"+opt.wp+'.pth'
        model.load_state_dict(torch.load(model_path))
        model.eval()
        if opt.gene!="jisin":
            #plot_code.save_npy_synthetic(model,validation_data,opt)
            plot_code.Compare_event_GT_pred(model,test_data,opt)
            plot_code.synthetic_plot(model,validation_data,opt)
        else:
            plot_code.Compare_event_GT_pred(model,test_data,opt)
        
        if opt.phase==True:
            plot_code.phase_eventGT_prediction_plot(model, test_data,opt)
        
        plot_code.near_tau_and_vector(model,opt)
        
        
    
#################
### Main
#################
def main():#python Main.py --pre_attn -imp=pre3_3 -trainvec_num=3 -pooling_k=3 --kmean --train --grad_log
    """ Main function. """#python Main.py --pre_attn -imp=pre3_3 -trainvec_num=3 -pooling_k=3 --kmean --train --grad_log
    parser = argparse.ArgumentParser()
    parser.add_argument('-epoch', type=int, default=1000)
    parser.add_argument('-batch_size', type=int, default=128)#32
    parser.add_argument('-loss_scale',type=int,default=1)

    parser.add_argument('-d_model', type=int, default=64)#512
    parser.add_argument('-d_inner_hid', type=int, default=64)#1024
    parser.add_argument('-d_k', type=int, default=8)#512
    parser.add_argument('-d_v', type=int, default=8)#512

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=4)
    parser.add_argument('-linear_num', type=int, default=3)
    

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-4)#1e-5
    parser.add_argument('-smooth', type=float, default=0.1)
    parser.add_argument('-gene', type=str, default='h1')
    parser.add_argument('-log', type=str, default='log/log.txt')
    
    parser.add_argument('-imp', type=str, default='_')
    parser.add_argument("-train_mean", type=float, default=0)
    parser.add_argument("-test_mean", type=float, default=0)

    parser.add_argument("-train_max", type=float, default=0)
    parser.add_argument("-train_min", type=float, default=0)
    parser.add_argument("-train_med", type=float, default=0)
    parser.add_argument("-time_step", type=int, default=30)
    parser.add_argument("-trainvec_num",type=int, default=3)
    parser.add_argument("-pooling_k",type=int, default=3)
    
    parser.add_argument("--train",action="store_true")
    parser.add_argument("--pickle_F",action='store_true')
    parser.add_argument("--pre_attn",action='store_true')
    parser.add_argument("--miman",action='store_true')

    parser.add_argument("-method",type=str, default="both_scalar")
    parser.add_argument("--kmean",action="store_true")
    parser.add_argument("--phase",action="store_true")
    parser.add_argument("--grad_log",action="store_true")
    parser.add_argument("-wp",type=str, default="_")
    parser.add_argument("-device_num",type=int, default=0)
    opt = parser.parse_args()

    # default device is CUDA
    opt.device = torch.device('cuda:'+str(opt.device_num))
    opt.log = "log/tau/"+str(opt.d_model)+'_'+str(opt.d_inner_hid)+'_'+str(opt.d_k)+'_'+str(opt.d_v)+"_"+str(opt.n_head)+'_'+opt.gene+'_'+opt.method+'_'+opt.imp+'_'+str(opt.epoch)+"_"+str(opt.time_step)+"_"+str(opt.trainvec_num)+"_"+str(opt.pooling_k)
    if opt.phase==True:
        opt.log+="_phase"
    if opt.pre_attn == True:
        opt.log+="_preLN"
    else:
        opt.log+="_postLN"
    opt.wp = opt.log
    opt.log+="_log.txt"
    print('[Info] parameters: {}'.format(opt))
    pickle_Flag=False
    """ prepare dataloader """
    if opt.gene=="jisin":
        df = pd.read_csv("data/date_jisin.90016")
        df["dtt64"] = pd.to_datetime(df["DateTime"])
        trainloader, validloader, testloader, opt.train_max ,opt.train_min, opt.train_med = set_date_class(df,opt)
        opt.gene = "jisin"
    
    elif opt.gene=="911_All":
        pickle_Flag= True
        train_path = "/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_All100_sliding_train.pkl"
        valid_path = "/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_All100_sliding_valid.pkl"
        test_path = "/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_All100_sliding_test.pkl"
    
    elif opt.gene=="911_1":
        pickle_Flag= True
        train_path = "/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_1_freq_sliding_train.pkl"
        valid_path = "/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_1_freq_sliding_valid.pkl"
        test_path = "/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_1_freq_sliding_test.pkl"

    elif opt.gene=="911_50":
        pickle_Flag= True
        train_path = "/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_50_freq_sliding_train.pkl"
        valid_path = "/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_50_freq_sliding_valid.pkl"
        test_path = "/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_50_freq_sliding_test.pkl"
    
    elif opt.gene=="911_100":
        pickle_Flag= True
        train_path = "/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_100_freq_sliding_train.pkl"
        valid_path = "/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_100_freq_sliding_valid.pkl"
        test_path = "/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_100_freq_sliding_test.pkl"

    else:
        trainloader, validloader, testloader, opt.train_max ,opt.train_min, opt.train_med = generate_data(opt.gene,opt)
    
    if pickle_Flag==True:
        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)
        with open(valid_path, 'rb') as f:
            valid_data = pickle.load(f)
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)
        opt.train_max=train_data.max()
        opt.train_min=train_data.min()
        opt.train_med=np.median(train_data)
        train_torch = torch.tensor(train_data).to(torch.double)
        valid_torch = torch.tensor(valid_data).to(torch.double) 
        test_torch = torch.tensor(test_data).to(torch.double) 
        trainloader = torch.utils.data.DataLoader(train_torch,num_workers=os.cpu_count(),batch_size=opt.batch_size,pin_memory=True,shuffle=True)
        validloader = torch.utils.data.DataLoader(valid_torch,num_workers=os.cpu_count(),batch_size=opt.batch_size,pin_memory=True,shuffle=False)
        testloader = torch.utils.data.DataLoader(test_torch,num_workers=os.cpu_count(),batch_size=opt.batch_size,pin_memory=True,shuffle=False)
    
    # setup the log filez
    
    if opt.train==True:
        with open(opt.log, 'w') as f:
            f.write('Loss,  Epoch,  Log-likelihood,  MAE,  RMSE\n')
    else:
        print("Not training")

    """ prepare model """
    model = Transformer(
        d_model=opt.d_model,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
        trainvec_num=opt.trainvec_num,
        pooling_k=opt.pooling_k,
        time_step=opt.time_step,
        device=opt.device,
        method=opt.method,
        train_max=opt.train_max,
        train_min=opt.train_min,
        train_med=opt.train_med,
        linear_layers=opt.linear_num,
        normalize_before=opt.pre_attn,
        train=trainloader,
        opt=opt
    )
    model.to(opt.device)
    if opt.train == True:
        with open(opt.log, 'a') as f:
            f.write('anchor values {anchor}\n'.format(anchor=model.anchor_vector))
            f.write('rep values {anchor}\n'.format(anchor=model.rep_vector))
    
        if opt.grad_log==True:
            os.makedirs('./param_grad/'+opt.imp, exist_ok = True)
            for name, param in model.named_parameters():#初期open
                with open("./param_grad/"+opt.imp+"/"+name+".log", 'w') as f:
                    f.write('st')
            
    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)


    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))
    
    """ train the model """
    if opt.phase==True:
        for param in model.parameters():
            param.requires_grad = True
        model.rep_vector.requires_grad = False
        model.anchor_vector.requires_grad = False
        """ optimizer and scheduler """
        optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05)
    
        scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)
        train(model, trainloader, validloader,testloader, optimizer, scheduler, opt)
        model_path="checkpoint/tau/"+opt.wp+'.pth'
        temp_path="checkpoint/tau/"+opt.gene+'_'+opt.method+'_'+opt.imp+'_'+str(opt.epoch)+"_"+str(opt.time_step)
        model.load_state_dict(torch.load(model_path))
        torch.save(model.state_dict(), opt.wp+"phase1.pth") 
        print("phase1 fin.")
        for param in model.parameters():
            param.requires_grad = True
        model.rep_vector.requires_grad = True
        model.anchor_vector.requires_grad = False
        """ optimizer and scheduler """
        optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05)
    
        scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)
        train(model, trainloader, validloader,testloader, optimizer, scheduler, opt)
        
        model.load_state_dict(torch.load(model_path))
        torch.save(model.state_dict(), opt.wp+"phase2.pth") 
        print("phase2 fin.")
        for param in model.parameters():
            param.requires_grad = True
        model.rep_vector.requires_grad = False
        model.anchor_vector.requires_grad = True
        """ optimizer and scheduler """
        optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)
        train(model, trainloader, validloader, testloader, optimizer, scheduler, opt)
        model.load_state_dict(torch.load(model_path))
        torch.save(model.state_dict(), opt.wp+"phase3.pth")
        print("phase2 fin.")
    else:
        train(model, trainloader, validloader, testloader, optimizer, scheduler, opt)

if __name__ == '__main__':
    plt.switch_backend('agg')
    plt.figure()
    main()