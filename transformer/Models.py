import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

import kmeans
import transformer.Constants as Constants
from transformer.Layers import EncoderLayer
from transformer.Layers import DecoderLayer
from matplotlib import pyplot as plt

def get_non_pad_mask(seq):
    """ Get the non-padding positions. """
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """
    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask

class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout,device,normalize_before,
            time_linear=False,train_max=0):
        super().__init__()
        
        self.d_model = d_model
        self.device=device
        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(train_max*1.5, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=device)
        self.normalize_before=normalize_before
        if normalize_before:
            self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
            for _ in range(n_layers)])
        self.train_max=train_max
    
    def temporal_enc(self, time):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        after_result = torch.zeros(result.shape,device=result.device)
        after_result[:,:, 0::2] = torch.sin(result[:, :,0::2])
        after_result[:, :, 1::2] = torch.cos(result[:, :,1::2])
        return after_result.to(result.device)

    def forward(self, event_time, rep_Mat=None,non_pad_mask=None):
        """ Encode event sequences via masked self-attention. """
        #入力の時間エンコーディング
        tem_enc = self.temporal_enc(event_time)
        tem_rep = self.temporal_enc(rep_Mat)#randomでない。バッチサイズにはなっている
        
        tem_enc = torch.cat((tem_enc,tem_rep), dim=1)#(16,seqence-1,M)->(16,seq-1+trainvec,M)
    
        if non_pad_mask is not None:
            slf_attn_mask_subseq = get_subsequent_mask(event_time)
            slf_attn_mask_subseq=torch.cat((torch.cat((slf_attn_mask_subseq,torch.zeros((event_time.shape[0],rep_Mat.shape[1],event_time.shape[1]),device=event_time.device)),dim=1),torch.zeros((event_time.shape[0],rep_Mat.shape[1]+event_time.shape[1],rep_Mat.shape[1]),device=event_time.device)),dim=2)
            slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=torch.cat((event_time,torch.ones((event_time.shape[0],rep_Mat.shape[1]),device=event_time.device)),dim=1), seq_q=torch.cat((event_time,torch.ones((event_time.shape[0],rep_Mat.shape[1]),device=event_time.device)),dim=1))
            slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
            slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        else:
            slf_attn_mask = None

        enc_output = torch.zeros(tem_enc.shape,device=self.device)
        for enc_layer in self.layer_stack:
            enc_output += tem_enc
            enc_output, _ = enc_layer(
                enc_input=enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            
        if self.normalize_before==True:
            enc_output = self.layer_norm(enc_output)
        
        return enc_output

class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim,linear_layers):
        super().__init__()
        
        layer_stack=[Linear_layers(int(dim/(2**i))) for i in range(linear_layers-1)]
        layer_stack.append(nn.Linear(int(dim/(2**(linear_layers-1))),1))
         
        self.layer_stack = nn.ModuleList(layer_stack)
        
        for i in range(len(self.layer_stack)-1):
            nn.init.xavier_uniform_(self.layer_stack[i].linear.weight)
        nn.init.xavier_uniform_(self.layer_stack[len(self.layer_stack)-1].weight)
    def forward(self, data):
        for linear_layer in self.layer_stack:
            data = linear_layer(data)
        return data    

class Decoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout,device,normalize_before,time_linear=False,train_max=0):
        super().__init__()

        self.d_model = d_model
        self.device = device
        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(train_max*1.5, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=device)
        
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
            for _ in range(n_layers)])
        
        self.train_max=train_max
    def temporal_enc(self, time):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        after_result = torch.zeros(result.shape,device=result.device)
        after_result[:,:, 0::2] = torch.sin(result[:, :,0::2])
        after_result[:, :, 1::2] = torch.cos(result[:, :,1::2])
        return after_result.to(result.device)
    def forward(self, input,k,v,temp_enc=True):
        #x:temp_enb(B,L,M)
        """ Encode event sequences via masked self-attention. """
        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        

        if temp_enc is True:
            x_tem_enc = self.temporal_enc(input)
        elif input.shape[2]==1:
            tem_parameter = input.repeat([k.shape[0],1,1])#repeat 引数は繰り返し回数 (1,vec_num)->(B,vec_num)
            x_tem_enc = self.temporal_enc(tem_parameter)
        else:
            x_tem_enc = input.repeat([k.shape[0],1,1])
        
        output = torch.zeros(x_tem_enc.shape,device=self.device)
        
        for dec_layer in self.layer_stack:
            output += x_tem_enc #residual
            output, _ = dec_layer(
                output,
                k,
                v,
                non_pad_mask=None,
                slf_attn_mask=None)
        return output


class Linear_layers(nn.Module):
    def __init__(
        self,d_model):
        super().__init__()
        self.linear = nn.Linear(d_model,int(d_model/2))
        self.relu = nn.ReLU()
    def forward(self,x):
        out = self.linear(x)
        out = self.relu(out)
        return out


    

class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            d_model=256, d_rnn=128, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1,trainvec_num=3,pooling_k=2,time_step=20,device="cuda:0",method="normal",train_max=0,train_min=0,train_med=0, linear_layers=1, normalize_before=True, train=None,opt=None):
        super().__init__()
        self.method = method
        
        self.encoder = Encoder(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            device=device,
            normalize_before=normalize_before,
            time_linear=False,
            train_max=train_max
        )
    
        self.decoder = Decoder(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            device=device,
            normalize_before=normalize_before,
            time_linear = False,
            train_max=train_max
        )
            
        
        # convert hidden vectors into a scalar
        layer_stack=[Linear_layers(int(d_model/(2**(i)))) for i in range(linear_layers-1)]
        layer_stack.append(nn.Linear( int(d_model/(2**(linear_layers-1))),1))
        self.layer_stack = nn.ModuleList(layer_stack)
        
        for i in range(len(self.layer_stack)-1):
            nn.init.xavier_uniform_(self.layer_stack[i].linear.weight)
        nn.init.xavier_uniform_(self.layer_stack[len(self.layer_stack)-1].weight)

        self.relu = nn.ReLU()
        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))
        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.rep_vector = nn.Parameter(torch.Tensor(kmeans.Set_data_kmeans(input=train, n_clusters=opt.trainvec_num)).to(opt.device).reshape((1, trainvec_num)))
        self.anchor_vector = nn.Parameter(torch.Tensor(kmeans.Set_data_kmeans(input=train, n_clusters=opt.pooling_k)).to(opt.device).reshape((1,pooling_k)))

        #self.rep_vector = nn.Parameter(torch.randn(1, trainvec_num))
        #self.anchor_vector = nn.Parameter(torch.randn(1,pooling_k))
        self.time_predictor = Predictor(d_model*pooling_k,linear_layers)
        self.rep_repeat=[]
        self.anc_repeat=[]
        
    
    def forward(self, input_time, target):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input:  event_time: batch*seq_len.
                Target:     batch*1.
        Output: enc_output: batch*(seq_len-1)*model_dim;
                time_prediction: batch*seq_len.
        """
        
        non_pad_mask = get_non_pad_mask(torch.cat((input_time,torch.ones((input_time.shape[0],self.rep_vector.shape[1]),device=input_time.device)),dim=1))
        rep_batch = self.rep_vector.repeat([input_time.shape[0],1])
        
        enc_output = self.encoder(input_time,rep_Mat=rep_batch,non_pad_mask=non_pad_mask) 
        enc_output = enc_output[:,-(rep_batch.shape[1]):,:]
        dec_output = self.decoder(target,k=enc_output,v=enc_output)
        
        anchor_batch = self.anchor_vector.repeat([enc_output.shape[0],1])
        time_hidden = self.decoder(anchor_batch,k=enc_output,v=enc_output)
        time_pred_decout_flatten = torch.flatten(time_hidden,1)
        time_prediction = self.time_predictor(time_pred_decout_flatten)
        return dec_output, time_prediction,enc_output
    