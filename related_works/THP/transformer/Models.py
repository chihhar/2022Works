import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

import transformer.Constants as Constants
from transformer.Layers import EncoderLayer
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
            n_layers, n_head, d_k, d_v, dropout,device,train_max,normalize_before):
        super().__init__()

        self.d_model = d_model
        self.device=device
        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(train_max*1.5, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=device)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
            for _ in range(n_layers)])
        self.train_max=train_max
        self.normalize_before=normalize_before
        if normalize_before:
            self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    def temporal_enc(self, time):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result

    def forward(self, event_time, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        if non_pad_mask is not None:
            slf_attn_mask_subseq = get_subsequent_mask(event_time)
            slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_time, seq_q=event_time)
            slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
            slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        else:
            slf_attn_mask = None

        tem_enc = self.temporal_enc(event_time)#入力の時間エンコーディング
            
        enc_output = torch.zeros(tem_enc.shape,device=self.device)
        """
        SS1=tem_enc[0,-2:-1,:]
        H0=tem_enc[0,:-1,:]
        
        for_i=0
        motonotime=[(self.train_max*1.5) / 1000 * i for i in range(1000)]
        tensortime=torch.tensor(motonotime).to(event_time.device).unsqueeze(0)
        temptime=self.temporal_enc(tensortime)
        """
        
        for enc_layer in self.layer_stack:
            enc_output += tem_enc
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        """ 
            if for_i==0:

                #S'r
                SD1=enc_output[0,-2:-1,:]
                SH1=enc_output[0,:-1,:]
            elif for_i==1:
                
                SDD1=enc_output[0,-2:-1,:]
                #S''
                SH2=enc_output[0,:-1,:]
            for_i+=1
            
        
        #enc_output (B,L-1,M)
        
        pdb.set_trace()
        S1_H0=torch.cosine_similarity(SS1,H0)
        SD1_H1=torch.cosine_similarity(SD1,SH1)
        SDD1_H2=torch.cosine_similarity(SDD1,SH2)
        plt.figure(figsize=(8,5))
        plt.plot(range(S1_H0.size(0)),S1_H0.cpu().detach(),label=r"initial");plt.plot(range(S1_H0.size(0)),SD1_H1.cpu().detach(),label=r"layer1");plt.plot(range(S1_H0.size(0)),SDD1_H2.cpu().detach(),label=r"layer2")
        plt.xlabel(r"past event index",fontsize=18)
        plt.ylabel("similarity",fontsize=18)
        plt.ylim(0.35,1.05)
        #plt.rc("svg", fonttype="none");
        plt.legend(fontsize=18)
        
        plt.savefig("plot/ronb/THP_Event_normDot_histi_.png")
        plt.savefig("plot/ronb/THP_Event_normDot_histi_.svg")
        plt.clf()
        pdb.set_trace()
        """
        #if self.normalize_before==True:
        #    enc_output = self.layer_norm(enc_output)
        return enc_output


class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim):
        super().__init__()
        
        #self.linear1 = nn.Linear(dim, dim, bias=False)#方式3
        #nn.init.xavier_normal_(self.linear1.weight)
        #self.relu = nn.ReLU()
        
        self.linear = nn.Linear(dim, 1, bias=False)
        nn.init.xavier_normal_(self.linear.weight)
        

    def forward(self, data):
        #data = self.linear1(data)
        #data = self.relu(data)#方式3

        return self.linear(data)
class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn):
        super().__init__()

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data, non_pad_mask):
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)
        
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

        out = self.projection(out)
        return out

class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            d_model=256, d_rnn=128, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1,time_step=20,device="cuda:0",train_max=0,normalize_before=True):
        super().__init__()

        self.encoder = Encoder(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            device=device,
            train_max=train_max,
            normalize_before=normalize_before
        )

        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, 1)
        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # OPTIONAL recurrent layer, this sometimes helps
        #self.rnn = RNN_layers(d_model, d_rnn)

        # prediction of next time stamp
        self.time_predictor = Predictor(d_model)
        #self.time_predictor = Predictor(d_model*3)
    def forward(self, input_time, target):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input:  event_time: batch*seq_len.
                Target:     batch*1.
        Output: enc_output: batch*(seq_len-1)*model_dim;
                time_prediction: batch*seq_len.
        """
        
        ##THP
        
        non_pad_mask = get_non_pad_mask(torch.cat((input_time,target),dim=1))#τ予測に真値が使われないようにするために必要。
        enc_output = self.encoder(torch.cat((input_time,target),dim=1), non_pad_mask=non_pad_mask)# 入力をエンコーダ部へ
        #enc_output = self.encoder(torch.cat((input_time,target),dim=1), non_pad_mask=None)
        
        #enc_output = self.rnn(enc_output, non_pad_mask=non_pad_mask)
        time_prediction = self.time_predictor(enc_output[:,-2:-1,:])
        
        #time_pred_decout_flatten = torch.flatten(enc_output[:,-4:-1,:],1)
        #time_prediction = self.time_predictor(time_pred_decout_flatten)#エンコーダの出力を線形変換によりτ予測　最後の行は未来を見ている。
        
        return enc_output, time_prediction[:,-1,:] #強度関数に必要な出力, 時間予測, エンコーダの出力
        #/THP
        