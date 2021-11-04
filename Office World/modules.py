'''
This script is taken from https://github.com/gblackoutwas4/NLIL.git
'''

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        sfm_attn = self.softmax(attn)

        output = torch.bmm(sfm_attn, v)

        return output, sfm_attn, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        '''
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        '''

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        
        '''
        nn.init.xavier_normal_(self.fc.weight)
        '''

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size() # (b, seq_len, d_in)
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, sfm_attn, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, sfm_attn, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, norm=True):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = output + residual
        if norm:
            output = self.layer_norm(output)

        return output


class FeedForwardConcat(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1, residual=True):
        super(FeedForwardConcat, self).__init__()
        self.w_1 = nn.Conv1d(d_in*2, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual

    def forward(self, x, y):

        # empty case
        if x.size(0) == 0:
            return x

        x, y = x.unsqueeze(0), y.unsqueeze(0)

        residual = x
        xy = torch.cat([x, y], dim=-1)

        output = xy.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)

        if self.residual:
            output = self.layer_norm(output + residual)
        else:
            output = self.layer_norm(output)

        return output.squeeze(0)


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None, norm=True):
        enc_output, _, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output, norm=norm)
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None, norm=True):
        dec_output, _, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        if non_pad_mask is not None:
            dec_output *= non_pad_mask

        dec_output, sfm_attn, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        if non_pad_mask is not None:
            dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output, norm=norm)
        if non_pad_mask is not None:
            dec_output *= non_pad_mask

        return dec_output, sfm_attn, dec_slf_attn, dec_enc_attn


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super().__init__()

        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                                          for _ in range(n_layers)])

    def forward(self, enc_input, enc_mask=None, return_attns=False, norm=True):

        enc_slf_attn_list = []
        slf_attn_mask, non_pad_mask = None, None

        # -- Prepare masks
        if enc_mask is not None:
            slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_mask, seq_q=enc_mask)
            non_pad_mask = get_non_pad_mask(enc_mask)

        # -- Forward
        enc_output = enc_input

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output,
                                                 non_pad_mask=non_pad_mask,
                                                 slf_attn_mask=slf_attn_mask,
                                                 norm=norm)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list

        return enc_output


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super().__init__()

        assert n_layers >= 1
        self.layer_stack = nn.ModuleList([DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                                          for _ in range(n_layers - 1)])

        # TODO last layer only single head maybe improve later?
        self.layer_stack.append(DecoderLayer(d_model, d_inner, 1, d_k, d_v, dropout=dropout))

    def forward(self, dec_input, enc_output, dec_mask=None, return_attns=False, norm=True):

        last_sfm_attn, last_dec_enc_attn = None, None
        non_pad_mask, slf_attn_mask = None, None

        # -- Forward
        dec_output = dec_input

        for dec_layer in self.layer_stack:
            dec_output, sfm_attn, dec_slf_attn, dec_enc_attn = dec_layer(dec_output, enc_output,
                                                                         non_pad_mask=non_pad_mask,
                                                                         slf_attn_mask=slf_attn_mask,
                                                                         dec_enc_attn_mask=dec_mask,
                                                                         norm=norm)

            if return_attns:
                last_sfm_attn = sfm_attn
                last_dec_enc_attn = dec_enc_attn

        if return_attns:
            return dec_output, last_sfm_attn, last_dec_enc_attn

        return dec_output


class Transformer(nn.Module):

    def __init__(self, steps=8,
                       inner=1100,
                       d_input=100,
                       output_embedding=False,
                       output_embedding_size=128,
                       dropout=0.1,
                       n_layers=3,
                       n_head=8):
        
        super().__init__()

        self.encoder = Encoder(n_layers=n_layers,
                               n_head=n_head,
                               d_k=d_input,
                               d_v=d_input,
                               d_model=d_input,
                               d_inner=output_embedding_size,
                               dropout=0.1)

        self.decoder = Decoder(n_layers=n_layers,
                               n_head=n_head,
                               d_k=d_input,
                               d_v=d_input,
                               d_model=d_input,
                               d_inner=output_embedding_size,
                               dropout=0.1)
        
        self.T = steps
        self.output_embedding = output_embedding
        if output_embedding:
            self.ffn = nn.Linear(inner, output_embedding_size)

    def forward(self, query):

        dec_output_list = []
        dec_output_embedding_list = []
        sfm_attn_list = []
        attn_list = []

        batch_size = query.shape[0]

        for _ in range(self.T):
            enc_output, dec_output, sfm_attn, attn = self.encode_decode(query)
            query = dec_output
            dec_output_list.append(dec_output)
            if self.output_embedding:
                dec_output_embedding_list.append(F.relu(self.ffn(dec_output.view(batch_size, -1))).view(batch_size, 1, -1))
            sfm_attn_list.append(sfm_attn)            
            attn = F.softmax(attn.sum(1), dim=-1)
            attn_list.append(attn.view(batch_size,-1,1))

        if  self.output_embedding:
            return dec_output_list, dec_output_embedding_list, sfm_attn_list, attn_list
        return dec_output_list, sfm_attn_list, attn_list

    def encode_decode(self, query, enc_output=None):
        '''
        :param query:
            (K, latent_dim)
        :return :
        '''

        enc_output = self.encoder(query)

        dec_output, sfm_attn, attn = self.decoder(query, enc_output, return_attns=True)

        return enc_output, dec_output, sfm_attn, attn
