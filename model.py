import torch
import torch.nn as nn 
# from Layers import EncoderLayer, DecoderLayer
# from Embed import Embedder, PositionalEncoder

import copy
import math
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.distributions as td

def AverageToken(X,mask_v):
    n, v, d = X.size()
    # Expand mask_v to have the same shape as X
    mask_v_expanded = mask_v.unsqueeze(-1).expand(n, v, d)  # shape: (n, v, d)
    # Compute the average values along the view dimension
    avg_token = torch.sum(X * mask_v_expanded, dim=1) / torch.sum(mask_v_expanded, dim=1)  # shape: (n, d)

    return avg_token
        
        
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def setEncoderModel(d_list,d_out):
    return nn.ModuleList([Mlp(d, 128, 2) for d in d_list])

def setDecoderModel(d_list,d_out):
    return nn.ModuleList([Mlp(512, 512, 512) for d in d_list])


def setEmbedingModel(d_list,d_out):
    
    return nn.ModuleList([Mlp(d,d,d_out)for d in d_list])

class Mlp(nn.Module):
    """ Transformer Feed-Forward Block """
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.5):
        super(Mlp, self).__init__()

        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)

        if self.dropout1:
            out = self.dropout1(out)
        out = self.fc2(out)
        if self.dropout1:
            out = self.dropout2(out)
        return out

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def attention(q, k, v, d_k, mask=None, dropout=None):

    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k) #scores shape is [bs heads view view]
    if mask is not None:
        mask = mask.unsqueeze(1).float()
        mask = mask.unsqueeze(-1).matmul(mask.unsqueeze(-2))#mask shape is [bs 1 view view]
        # mask = mask.unsqueeze(1) #mask shape is [bs 1 1 view]
        scores = scores.masked_fill(mask == 0, -1e9)
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.2):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):

        bs = q.size(0)
        
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * d_model/h
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        output = self.out(concat)
    
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.2):
        super().__init__() 
    
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout_2 = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.dropout_1(F.relu(self.linear_1(x)))
        x = self.dropout_2(self.linear_2(x))
        return x
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.2):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)

        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    
# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, \
        src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x
class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        print(x)
        return self.embed(x)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 200, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        pe = Parameter(self.pe[:,:seq_len])
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        # self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        x = src
        # x = self.embed(src)
        # x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        # self.embed = Embedder(vocab_size, d_model)
        # self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = trg
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        #print("DECODER")
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output

class TransformerWoDecoder(nn.Module):
    def __init__(self, src_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        # self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        # self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, src_mask=None):
        e_outputs = self.encoder(src, src_mask)
        #print("DECODER")
        # d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        # output = self.out(e_outputs)
        return e_outputs


class encoder(nn.Module):
    def __init__(self, n_dim, dims, n_z,dropout):
        super(encoder, self).__init__()
        # print(n_dim,dims[0])
        self.enc_1 = nn.Linear(n_dim, dims[0])
        self.enc_2 = nn.Linear(dims[0], dims[1])
        self.enc_3 = nn.Linear(dims[1], dims[2])
        self.z_layer = nn.Linear(dims[2], n_z)
        self.z_b0 = nn.BatchNorm1d(n_z)
        self.dropout=nn.Dropout(dropout)
    def forward(self, x):
        enc_h1 = self.dropout(F.relu(self.enc_1(x)))
        enc_h2 = self.dropout(F.relu(self.enc_2(enc_h1)))
        enc_h3 = self.dropout(F.relu(self.enc_3(enc_h2)))
        z = self.z_b0(self.z_layer(enc_h3))
        return z

class decoder(nn.Module):
    def __init__(self, n_dim, dims, n_z,dropout):
        super(decoder, self).__init__()
        self.dec_0 = nn.Linear(n_z, n_z)
        self.dec_1 = nn.Linear(n_z, dims[2])
        self.dec_2 = nn.Linear(dims[2], dims[1])
        self.dec_3 = nn.Linear(dims[1], dims[0])
        self.x_bar_layer = nn.Linear(dims[0], n_dim)
        self.dropout=nn.Dropout(dropout)
    def forward(self, z):
        r = F.relu(self.dec_0(z))
        # dec_h_try = F.relu(self.dec_1(r)).detach().cpu().numpy()
        dec_h1 = self.dropout(F.relu(self.dec_1(r)))
        # dech_h1_try = dec_h1.detach().cpu().numpy()
        dec_h2 = self.dropout(F.relu(self.dec_2(dec_h1)))
        dec_h3 = self.dropout(F.relu(self.dec_3(dec_h2)))
        x_bar = self.x_bar_layer(dec_h3)
        return x_bar

class Model(nn.Module):
    def __init__(self, input_len, d_model, n_layers, heads, d_list, classes_num, dropout,exponent=2):
        super().__init__()
        self.Trans = TransformerWoDecoder(input_len, d_model, 6, heads, dropout)
        self.d_model = d_model
        #self.Mixer = Encoder2(d_model, input_len, n_layers, dropout)
        #self.CMixer = Encoder2(d_model, classes_num+1, 1,dropout)
        self.embeddinglayers = setEmbedingModel(d_list,d_model)
        self.view_num = input_len
        #self.ae = AE(n_stacks=4,n_input=d_list,n_z=d_model,nLabel=classes_num,dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.CFTrans = TransformerWoDecoder(classes_num+1, d_model, 2, heads, dropout)
        self.classifiers = nn.ModuleList([nn.Linear(d_model,1)for _ in range(classes_num)])
        self.classifier2 = nn.Linear(512,classes_num)
        self.adaptive_weighting = nn.Linear(d_model,1,bias=False)
        self.weights = Parameter(torch.softmax(torch.randn([1,self.view_num,1]),dim=1))
        #self.weights = torch.full([1, self.view_num, 1], 1.0 / self.view_num)
        self.exponent = exponent
        self.cls_tokens = nn.Parameter(torch.randn(1, classes_num, d_model))
        self.confidence = nn.Linear(d_model,1)
        torch.nn.init.xavier_uniform_(self.cls_tokens,gain=1)
        
    def forward(self,x,mask=None,label_mask = None):
        B = mask.shape[0]
        x_res = [x[0],x[1],x[2],x[3],x[4],x[5]]
        for i in range(self.view_num):
            x[i] = self.embeddinglayers[i](x[i])
        x = torch.stack(x,dim=1) # B,view,d

        
        AVT = AverageToken(x,mask)
        AVT= AVT.unsqueeze(1).expand(B, 3, self.d_model)
        x = torch.cat((x, AVT), dim=1)

        ones_column = torch.ones((B, 3), dtype=mask.dtype, device=mask.device)

        mask2 = torch.cat((mask, ones_column), dim=1)

        x = self.Trans(x,mask2)
        x = x[:,:self.view_num,:]
        
        x_tran = x

        x_weighted = torch.pow(self.weights.expand(B,-1,-1),self.exponent)

        x_weighted_mask = torch.softmax(x_weighted.masked_fill(mask.unsqueeze(2)==0, -1e9),dim=1) #[B, self.view_num, 1]
        assert torch.sum(torch.isnan(x_weighted_mask)).item() == 0
        # print('mask',mask[:3,:])
        x = x.mul(x_weighted_mask)
        fusion_x = torch.einsum('bvd->bd',x)

        cls_tokens = self.cls_tokens.expand(B,-1,-1).to(fusion_x.device)
        x_tokens = torch.concat((fusion_x.unsqueeze(1),cls_tokens),dim=1)
        if label_mask is not None:
            label_mask = torch.concat((torch.ones([B,1],device=label_mask.device),label_mask),dim=1)  
        else: 
            label_mask = None
        x_tokens = self.CFTrans(x_tokens)
        pred2 = []
        pred = torch.sigmoid(self.classifier2(x_tokens[:,0]))
        confidence=1

        return[pred,pred2],1,x_tran,cls_tokens


def get_model(input_len, d_list,d_model=768,n_layers=2,heads=4,classes_num=10,dropout=0.2,exponent=1,load_weights=None,device=torch.device('cuda:0')):
    
    assert d_model % heads == 0
    assert dropout < 1

    model = Model(input_len, d_model, n_layers, heads, d_list, classes_num, dropout, exponent)

    if load_weights is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
    else:
        for p in model.parameters():
            if p is not model.weights and p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    
    
    model = model.to(device)
    
    return model
    
