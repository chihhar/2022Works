import torch.nn as nn

from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)
        self.normalize_before=normalize_before
    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):      
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, mask=slf_attn_mask)##q=k=v
        if non_pad_mask is None:
            enc_output = self.pos_ffn(enc_output)
        else:
            enc_output *= non_pad_mask
            enc_output = self.pos_ffn(enc_output)
            enc_output *= non_pad_mask
        
        return enc_output, enc_slf_attn
class DecoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, dec_input, k, v, non_pad_mask=None, slf_attn_mask=None):
        #enc_output, enc_slf_attn = self.slf_attn(
        #    enc_input, enc_input, enc_input, mask=slf_attn_mask)##q=k=v

        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, k, v, mask=slf_attn_mask)##q!=k=v
        if non_pad_mask is None:
            dec_output = self.pos_ffn(dec_output)
        else:
            dec_output *= non_pad_mask
            dec_output = self.pos_ffn(dec_output)
            dec_output *= non_pad_mask
        
        return dec_output, dec_slf_attn