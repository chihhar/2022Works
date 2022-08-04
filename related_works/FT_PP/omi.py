import pandas as pd
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
from distutils.util import strtobool
import os

import tensorflow as tf

import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

import pdb
parser = argparse.ArgumentParser()
parser.add_argument("--train",type=str, default='False', choices=['True', 'False'])
parser.add_argument("-gene",type=str,default="")
parser.add_argument("-model",type=str, default='omi', choices=['const','exp','pc','omi'])
parser.add_argument("-t_max",default=0.0)
args = parser.parse_args()

isTrain = strtobool(args.train)
data_dir = './'
visual_dir = 'visualization'
checkpoint_dir = "checkpoint/"+args.model+'/'+args.gene+"_checkpoint"
checkpoint_path = f"{checkpoint_dir}/omi.ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#----------------------------

#----------------------------
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

def generate_hawkes1():
    np.random.seed(seed=32)
    [T,LL] = simulate_hawkes(100000,0.2,[0.8,0.0],[1.0,20.0])
    score = - LL[80000:].mean()
    return T
def generate_hawkes_modes():
    np.random.seed(seed=32)
    [T,LL,L_TRG1] = simulate_hawkes_modes(100000,0.2,[0.8,0.0],[1.0,20.0])
    score = - LL[80000:].mean()
    return T

def simulate_hawkes_modes(n,mu,alpha,beta,short_thre=1,long_thre=5):
    T = []
    LL = []
    L_TRG1 = []
    
    x = 0
    l_trg1 = 0
    l_trg2 = 0
    l_trg_Int1 = 0
    l_trg_Int2 = 0
    mu_Int = 0
    count = 0
    is_long_mode = 0
    
    while 1:
        l = mu + l_trg1 + l_trg2
        #step = np.random.exponential(scale=1)/l

        if l_trg1 > long_thre:
            is_long_mode = 1

        if l_trg1 < short_thre:
            is_long_mode = 0

        if is_long_mode: # long mode
            step = step = np.random.exponential(scale=2)/l
        else: # short mode
            step = np.random.exponential(scale=0.5)/l

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
            L_TRG1.append(l_trg1)
            l_trg1 += alpha[0]*beta[0]
            l_trg2 += alpha[1]*beta[1]
            l_trg_Int1 = 0
            l_trg_Int2 = 0
            mu_Int = 0
            count += 1
        
        if count == n:
            break
        
    return [np.array(T),np.array(LL),np.array(L_TRG1)]

def generate_hawkes_modes05():
    np.random.seed(seed=32)
    [T,LL,L_TRG1] = simulate_hawkes_modes05(100000,0.2,[0.8,0.0],[1.0,20.0])
    score = - LL[80000:].mean()
    return T

def simulate_hawkes_modes05(n,mu,alpha,beta,short_thre=1,long_thre=5):
    T = []
    LL = []
    L_TRG1 = []
    
    x = 0
    l_trg1 = 0
    l_trg2 = 0
    l_trg_Int1 = 0
    l_trg_Int2 = 0
    mu_Int = 0
    count = 0
    is_long_mode = 0
    
    while 1:
        l = mu + l_trg1 + l_trg2
        #step = np.random.exponential(scale=1)/l

        if l_trg1 > long_thre:
            is_long_mode = 1

        if l_trg1 < short_thre:
            is_long_mode = 0

        if is_long_mode: # long mode
            step = step = np.random.exponential(scale=2)/l
        else: # short mode
            step = np.random.exponential(scale=0.25)/l

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
            L_TRG1.append(l_trg1)
            l_trg1 += alpha[0]*beta[0]
            l_trg2 += alpha[1]*beta[1]
            l_trg_Int1 = 0
            l_trg_Int2 = 0
            mu_Int = 0
            count += 1
        
        if count == n:
            break
        
    return [np.array(T),np.array(LL),np.array(L_TRG1)]
    
def generate_jisin(data_dir):
    df = pd.read_csv("/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/date_jisin.90016")
    df["dt64"] = pd.to_datetime(df["DateTime"])
    df["dt64"] = df["dt64"].map(pd.Timestamp.timestamp)/3600 # UNIX変換
    T=df["dt64"]
    return T.values

def rolling_matrix(x,window_size):
    x = x.flatten()
    n = x.shape[0]
    stride = x.strides[0]
    return np.lib.stride_tricks.as_strided(x, shape=(n-window_size+1, window_size), strides=(stride,stride) ).copy()

def transform_data(T,n_train,n_validation,n_test,time_step):
    np.random.seed(0)
    index_shuffle = np.random.permutation(n_train-time_step)

    dT_train = np.ediff1d(T[:n_train])
    r_dT_train = rolling_matrix(dT_train,time_step)[index_shuffle]
    
    dT_valid = np.ediff1d(T[n_train:n_train+n_validation])
    r_dT_valid = rolling_matrix(dT_valid,time_step)

    dT_test = np.ediff1d(T[n_train+n_validation:n_train+n_validation+n_test])
    r_dT_test = rolling_matrix(dT_test,time_step)

    dT_train_input  = r_dT_train[:,:-1].reshape(-1,time_step-1,1)
    dT_train_target = r_dT_train[:,[-1]]
    dT_valid_input  = r_dT_valid[:,:-1].reshape(-1,time_step-1,1)
    dT_valid_target = r_dT_valid[:,[-1]]
    dT_test_input  = r_dT_test[:,:-1].reshape(-1,time_step-1,1)
    dT_test_target = r_dT_test[:,[-1]]
    args.t_max=dT_train_target.max()*1.001
    return [dT_train_input,dT_train_target,dT_valid_input,dT_valid_target,dT_test_input,dT_test_target]
#----------------------------

#----------------------------
def negative_log_likelihood(log_l, y_pred):
    # loss = -(sum_hazard - cum_hazard)
    loss = -(log_l - y_pred)
    return loss

def log(anti, delta=1e-10):
    logar = tf.math.log(tf.clip_by_value(anti, clip_value_min=delta, clip_value_max=K.max(anti)))
    return logar

# kernel initializer for positive weights
def abs_glorot_uniform(shape, dtype=None, partition_info=None):
    return K.abs(keras.initializers.glorot_uniform(seed=None)(shape,dtype=dtype))

######################################################
### constant hazard function
######################################################
class HAZARD_const(tf.keras.layers.Layer):

    def __init__(self):
        super(HAZARD_const, self).__init__()
        self.const_dense=layers.Dense(1)

    def call(self, future, past_hidden):
        x = future
        p = self.const_dense(past_hidden)
        log_l = p
        Int_l = K.exp( p ) * x        
        return log_l, Int_l

######################################################
### exponential hazard function
######################################################
class HAZARD_exp(tf.keras.layers.Layer):
    

    def __init__(self):
        super(HAZARD_exp, self).__init__()
        self.a = self.add_weight(name='a', initializer= keras.initializers.Constant(value=1.0), shape=(), trainable=True)
        self.exp_dense=layers.Dense(1)
    def call(self, future, past_hidden):
        x = future
        p = self.exp_dense(past_hidden)
        a = self.a
        log_l = p - a*x
        Int_l = K.exp( p ) * ( 1 - K.exp(-a*x) ) / a
        return log_l, Int_l

######################################################
### piecewise constant hazard function
######################################################   
class HAZARD_pc(tf.keras.layers.Layer):
    def __init__(self,size_div,t_max):
        super(HAZARD_pc, self).__init__()

        self.size_div = size_div
        self.t_max = t_max
    
        self.bin_l = K.constant(np.linspace(0,t_max,size_div+1)[:-1].reshape(1,-1))
        self.bin_r = K.constant(np.linspace(0,t_max,size_div+1)[1:].reshape(1,-1))
        self.width = K.constant(t_max/size_div)
        self.ones = K.constant(np.ones([size_div,1]))
        self.pc_dense = layers.Dense(self.size_div,activation='softplus')
    def call(self, future, past_hidden):
        x = future
        p = self.pc_dense(past_hidden)
        r_le = K.cast( K.greater_equal(x,self.bin_l), dtype=K.floatx() ) 
        r_er = K.cast( K.less(         x,self.bin_r), dtype=K.floatx() )
        r_e  = r_er*r_le
        r_l  = 1-r_er
        
        log_l = K.log(K.dot(p*r_e,self.ones))
        Int_l = K.dot(p*r_l*self.width,self.ones) + K.dot(p*(x-self.bin_l)*r_e,self.ones) 

        return log_l, Int_l

class cumlative_hazard_function(tf.keras.layers.Layer):
    def __init__(self, nn_size=64, size_layer_chfn=2):
        super(cumlative_hazard_function, self).__init__()

        self.fc_future = layers.Dense(units=nn_size, name='chf', kernel_initializer=abs_glorot_uniform, kernel_constraint=keras.constraints.NonNeg(), use_bias=False)

        self.fcs = [ layers.Dense(units=nn_size, name='chf', kernel_initializer=abs_glorot_uniform, kernel_constraint=keras.constraints.NonNeg(), activation='tanh') for _ in range(size_layer_chfn-1) ]
        
        self.fcs_tau = [ layers.Dense(units=nn_size, name='chf_tau',  kernel_initializer=abs_glorot_uniform, kernel_constraint=keras.constraints.NonNeg(), activation='tanh') for _ in range(size_layer_chfn-1) ]

        self.fcs_mark = [ layers.Dense(units=nn_size, name='chf_mark',  kernel_initializer=abs_glorot_uniform, kernel_constraint=keras.constraints.NonNeg(), activation='tanh') for _ in range(size_layer_chfn-1) ]

        self.fc_tau = layers.Dense(1, name='chf_tau', kernel_initializer=abs_glorot_uniform, kernel_constraint=keras.constraints.NonNeg(), activation='softplus')
        self.fc_mark = layers.Dense(1, name='chf_mark', kernel_initializer=abs_glorot_uniform, kernel_constraint=keras.constraints.NonNeg(), activation='softplus')
        self.fc = layers.Dense(1, name='chf', kernel_initializer=abs_glorot_uniform, kernel_constraint=keras.constraints.NonNeg())

    def call(self, future, past_hidden):
        future_hidden = self.fc_future(future)

        combine_hidden = future_hidden + past_hidden
        combine_hidden = K.tanh(combine_hidden)        

        ### fusion network
        for i in range(len(self.fcs)):
            combine_hidden = self.fcs[i](combine_hidden)
        intermediate_var_s = K.tanh(combine_hidden)

        chf_tau = self.fc_tau(intermediate_var_s)
        return None, chf_tau

class omi(tf.keras.Model):
    #-------
    def __init__(self, visual_path='visualization', type_hazard='omi',baseChn=16, temp_feature_size=64, nn_size=64, size_div=128, size_layer_chfn=2, hidden_num=30, isNormalize=True):
        super(omi, self).__init__()

        self.visual_path = visual_path
        self.baseChn = baseChn
        self.temp_feature_size = temp_feature_size
        self.nn_size = nn_size
        self.size_layer_chfn = size_layer_chfn
        self.hidden_num = hidden_num
        self.isNormalize = isNormalize
        self.size_div=size_div
        self.type_hazard=type_hazard
        self.t_max=args.t_max
        self.T_mean = 0
        self.T_std = 1

        # expand mark and temp
        self.fc_past_temp = layers.TimeDistributed(layers.Dense(units=temp_feature_size, name='lstm'))
        # extract future of past sequence
        self.lstm_temp = layers.LSTM(temp_feature_size, name='lstm_tau', activation='tanh')
    #-------

    #-------
    def build(self, input_shape):
        # cumulative hazard function
        if self.type_hazard == 'const':
            self.layer_hazard = HAZARD_const()
        elif self.type_hazard == 'exp':
            self.layer_hazard = HAZARD_exp()
        elif self.type_hazard == 'pc':
            self.layer_hazard = HAZARD_pc(size_div=self.size_div,t_max=self.t_max)
        elif self.type_hazard == 'omi':
            self.layer_hazard = cumlative_hazard_function(nn_size=self.nn_size,  size_layer_chfn=self.size_layer_chfn)
    #-------

    #-------
    def call(self, x):
        diff_T_past, diff_T_future = x[0]
        diff_T_future = tf.Variable(diff_T_future)

        if self.isNormalize & (self.type_hazard=='omi'):
            diff_T_past_nmlz = (diff_T_past-self.T_mean)/self.T_std
            diff_T_future_nmlz = (diff_T_future-self.T_mean)/self.T_std
        else:
            diff_T_past_nmlz = diff_T_past
            diff_T_future_nmlz = diff_T_future
        
        #---------
        # extract feature of past sequence
        diff_T_past = self.fc_past_temp(diff_T_past)
        past_hidden_fc = self.lstm_temp(diff_T_past)
        #---------
        
        #---------
        # cumulative hazard function
        hazard_tau, cum_hazard_tau = self.layer_hazard(diff_T_future_nmlz, past_hidden_fc)
        #---------

        return hazard_tau, cum_hazard_tau, diff_T_future

    # train step
    def train_step(self, x):

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2, tf.GradientTape() as tape3, tf.GradientTape() as tape4:
            
            #predict
            hazard_tau, cum_hazard_tau, diff_T_future = self(x, training=True)
            if self.type_hazard == 'omi':
                tape1.watch(diff_T_future)

                # hazard function and its log-likelihood
                hazard_tau = tape1.gradient(cum_hazard_tau, diff_T_future)
                
                # if np.sum(hazard_tau <= 0):
                #    pdb.set_trace
                sum_hazard = log(hazard_tau)
            else:
                sum_hazard=hazard_tau
            # train using gradients
            trainable_vars = self.trainable_variables

            log_l = sum_hazard
            y_pred = cum_hazard_tau
            trainable_vars = [v for v in trainable_vars if 'lstm' in v.name or 'chf' in v.name or 'transformerpp' in v.name]

            loss = self.compiled_loss(log_l, y_pred, regularization_losses=self.losses)

            gradients = tape3.gradient(loss, trainable_vars)
            # if np.sum([np.sum(np.isnan(gradients[i])) for i in range(len(gradients))]):
            #     pdb.set_trace()

        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        self.optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, trainable_vars) if grad is not None)

        # update metrics
        self.compiled_metrics.update_state(log(hazard_tau + 1e-10), cum_hazard_tau)

        # return metrics as dictionary
        return {m.name: m.result() for m in self.metrics}
    #-------

    #-------
    # test step
    def test_step(self, x):

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:

            #predict
            hazard_tau, cum_hazard_tau, diff_T_future = self(x, training=False)
            if self.type_hazard=='omi':
                tape1.watch(diff_T_future)
                
                # hazard function and its log-likelihood
                hazard_tau = tape1.gradient(cum_hazard_tau, diff_T_future)
            
                sum_hazard = log(hazard_tau)
            else:
                sum_hazard=hazard_tau
            log_l = sum_hazard
            y_pred = cum_hazard_tau

        # loss
        self.compiled_loss(log_l, y_pred, regularization_losses=self.losses)

        self.compiled_metrics.update_state(log(hazard_tau), cum_hazard_tau)

        # return metrics as dictionary
        return {m.name: m.result() for m in self.metrics}
    #-------

    #-------
    # predict step
    def predict_step(self, x):

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            
            #predict
            hazard_tau, cum_hazard_tau, diff_T_future = self(x, training=False)
            tape1.watch(diff_T_future)
            if self.type_hazard=='omi':
                # hazard function and its log-likelihood
                hazard_tau = tape1.gradient(cum_hazard_tau, diff_T_future)
                sum_hazard = log(hazard_tau)
            else:
                sum_hazard=hazard_tau
            

        return cum_hazard_tau, hazard_tau
#----------------------------

#----------------------------
def plotT(true_hist, pred_hist, label='FT-PP'):
    plt.figure(figsize=(10,4))
    plt.xlabel("event ID", fontsize=18)
    plt.ylabel("elapsed time", fontsize=18)
    plt.plot(range(100),true_hist[0:100],label="ground-truth")
    plt.plot(range(100),pred_hist[0:100],c="r",label=f"{label}",linestyle="dashed")
    plt.legend(fontsize=18, loc='upper right')
    plt.savefig(f'{visual_dir}/{args.gene}_{args.model}_512_tau_{label}.pdf', bbox_inches='tight', pad_inches=0)
#----------------------------

#----------------------------
def main():
    pickle_Flag=False
    # データの読み込み
    if args.gene == "jisin":
        T = generate_jisin(data_dir)
    elif args.gene=="h1":
        T = generate_hawkes1()
    elif args.gene=='h_fix':
        T = generate_hawkes_modes()
    elif args.gene=='h_fix05':
        T = generate_hawkes_modes05()
    elif args.gene=="911_All":
        pickle_Flag= True
        train_path = "/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_All100_sliding_train.pkl"
        valid_path = "/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_All100_sliding_valid.pkl"
        test_path = "/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_All100_sliding_test.pkl"
    
    elif args.gene=="911_1":
        pickle_Flag= True
        train_path = "/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_1_freq_sliding_train.pkl"
        valid_path = "/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_1_freq_sliding_valid.pkl"
        test_path = "/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_1_freq_sliding_test.pkl"

    elif args.gene=="911_50":
        pickle_Flag= True
        train_path = "/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_50_freq_sliding_train.pkl"
        valid_path = "/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_50_freq_sliding_valid.pkl"
        test_path = "/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_50_freq_sliding_test.pkl"
    
    elif args.gene=="911_100":
        pickle_Flag= True
        train_path = "/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_100_freq_sliding_train.pkl"
        valid_path = "/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_100_freq_sliding_valid.pkl"
        test_path = "/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_100_freq_sliding_test.pkl"
    else:
        print("opt.gene is nodata")
    if pickle_Flag==True:
        time_step=30
        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)
        with open(valid_path, 'rb') as f:
            valid_data = pickle.load(f)
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)
        dT_train_input  = train_data[:,:-1,np.newaxis]
        dT_train_target = train_data[:,[-1]]
        dT_valid_input  = valid_data[:,:-1,np.newaxis]
        dT_valid_target = valid_data[:,[-1]]
        dT_test_input  = test_data[:,:-1,np.newaxis]
        dT_test_target = test_data[:,[-1]]
        data = [dT_train_input,dT_train_target,dT_valid_input,dT_valid_target,dT_test_input,dT_test_target]
        args.t_max=data[1].max()*1.001
    else:
        #T = generate_jisin(data_dir) if isEarthquake else generate_hawkes1()
        # 前処理
        data = transform_data(T, int(len(T)*0.8), int(len(T)*0.1),int(len(T)*0.1), 30)
    # モデル形成
    model = omi(type_hazard=args.model,visual_path='visualization')
    model.T_mean = np.mean(data[0])
    model.T_std = np.std(data[0])
    #model.load_weights(checkpoint_path)
    pp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_weights_only=True, save_best_only=True, verbose=1)
    cp_early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='auto', min_delta=0, verbose=1)
    model.compile(optimizer='adam', loss=negative_log_likelihood, metrics=[negative_log_likelihood], run_eagerly=True)
    #pdb.set_trace()

    if isTrain==True:
        #pdb.set_trace()
        print("train")
        history = model.fit((data[0], data[1]), batch_size=64, epochs=1000, validation_split=0.1, callbacks=[cp_early_stopping, pp_callback])
    else:
        #pdb.set_trace()
        print("load")
        model.load_weights(checkpoint_path)
    
    #model.load_weights(checkpoint_path)
    # 予測 
    # The median of the predictive distribution is determined using the bisection method.
    x_left = 1e-4  * np.mean(data[0]) * np.ones_like(data[3])
    x_right = 100 * np.mean(data[0]) * np.ones_like(data[3])
    for i in range(13):
        x_center = (x_left+x_right)/2
        v = model.predict((data[2], x_center), batch_size=x_center.shape[0])[0]
        x_left = np.where(v<np.log(2),x_center,x_left)
        x_right = np.where(v>=np.log(2),x_center,x_right)
    valid_tau_pred = (x_left+x_right)/2 # predicted interevent interval
    valid_AE = np.abs(data[3]-valid_tau_pred) # absolute error
    
    x_left = 1e-4  * np.mean(data[0]) * np.ones_like(data[5])
    x_right = 100 * np.mean(data[0]) * np.ones_like(data[5])
    for i in range(13):
        x_center = (x_left+x_right)/2
        v = model.predict((data[4], x_center), batch_size=x_center.shape[0])[0]
        x_left = np.where(v<np.log(2),x_center,x_left)
        x_right = np.where(v>=np.log(2),x_center,x_right)
    test_tau_pred = (x_left+x_right)/2 # predicted interevent interval
    test_AE = np.abs(data[5]-test_tau_pred) # absolute error
    #pdb.set_trace()
    print("valid Mean absolute error: ", valid_AE.mean() )
    print("test Mean absolute error: ", test_AE.mean() )
    plotT(data[5], test_tau_pred)
    np.save(f'{checkpoint_dir}/tau_pred.npy', test_tau_pred)
    #pdb.set_trace()

if __name__ == '__main__':
    main()