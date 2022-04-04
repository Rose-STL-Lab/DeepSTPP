import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

"""
Return a square attention mask to only allow self-attention layers to attend the earlier positions
"""
def subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


"""
Injects some information about the relative or absolute position of the tokens in the sequence
ref: https://github.com/harvardnlp/annotated-transformer/
"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

    def forward(self, x, t):
        pe = torch.zeros(self.max_len, self.d_model).cuda()
        
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)).cuda()
        
        t = t.unsqueeze(-1)
        pe = torch.zeros(*t.shape[:2], self.d_model).cuda()
        pe[..., 0::2] = torch.sin(t * div_term)
        pe[..., 1::2] = torch.cos(t * div_term)
        
        x = x + pe[:x.size(0)]
        return self.dropout(x)
    
    
"""
Encode time/space record to variational posterior for location latent
"""
class Encoder(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(config.emb_dim, config.dropout, 
                                              config.seq_len)
        encoder_layers = nn.TransformerEncoderLayer(config.emb_dim, config.num_head,
                                                    config.hid_dim, config.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, config.nlayers)
        self.seq_len = config.seq_len
        self.ninp = config.emb_dim
        self.encoder = nn.Linear(3, config.emb_dim, bias=False)
        self.decoder = nn.Linear(config.emb_dim, config.z_dim * 2)
        self.init_weights()
        self.device = device
        
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def encode(self, x, x_mask=None):
        x = x.transpose(1,0) # Convert to seq-len-first
        if x_mask is None:
            x_mask = subsequent_mask(len(x)).to(self.device)
        t = torch.cumsum(x[..., -1], 0)
        x = self.encoder(x) * math.sqrt(self.ninp)
        x = self.pos_encoder(x, t)
        
        output = self.transformer_encoder(x, x_mask)
        output = self.decoder(output)
        
        output = output[-1] # get last output only
        m, v_ = torch.split(output, output.size(-1) // 2, dim=-1)
        v = F.softplus(v_) + 1e-5
        return m, v
    
"""
Decode latent variable to spatiotemporal kernel coefficients
"""
class Decoder(nn.Module):
    def __init__(self, config, out_dim, softplus=False):
        super().__init__()
        self.z_dim = config.z_dim
        self.softplus = softplus
        self.net = nn.Sequential(
            nn.Linear(config.z_dim, config.hid_dim),
            nn.ELU(),
            nn.Linear(config.hid_dim, config.hid_dim),
            nn.ELU(),
            nn.Linear(config.hid_dim, config.hid_dim),
            nn.ELU(),
            nn.Linear(config.hid_dim, out_dim),
        )

    def decode(self, z):
        output = self.net(z)
        if self.softplus:
            output = F.softplus(output) + 1e-5
        return output
    
    
def kl_normal(qm, qv, pm, pv):
    """
    Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
    sum over the last dimension

    Args:
        qm: tensor: (batch, dim): q mean
        qv: tensor: (batch, dim): q variance
        pm: tensor: (batch, dim): p mean
        pv: tensor: (batch, dim): p variance

    Return:
        kl: tensor: (batch,): kl between each sample
    """
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    kl = element_wise.sum(-1)
    return kl


def sample_gaussian(m, v):
    """
    Element-wise application reparameterization trick to sample from Gaussian

    Args:
        m: tensor: (batch, ...): Mean
        v: tensor: (batch, ...): Variance

    Return:
        z: tensor: (batch, ...): Samples
    """
    z = torch.randn_like(m)
    z = z * torch.sqrt(v) + m
    return z


"""
Log likelihood of no events happening from t_n to t
- ∫_{t_n}^t λ(t') dt' 

tn_ti: [batch, seq_len]
t_ti: [batch, seq_len]
w_i: [batch, seq_len]
b_i: [batch, seq_len]

return: scalar
"""
def ll_no_events(w_i, b_i, g_i, tn_ti, t_ti):
    return np.sqrt(np.pi / 2) * w_i * (torch.erf(b_i - torch.log(t_ti) / np.sqrt(2))
    - torch.erf(b_i - torch.log(tn_ti) / np.sqrt(2)))
    #return torch.sum(w_i / b_i * (torch.exp(-b_i * t_ti) - torch.exp(-b_i * tn_ti)), -1)


def log_ft(t_ti, tn_ti, w_i, b_i, g_i):
    return ll_no_events(w_i, b_i, g_i, tn_ti, t_ti) 
    + torch.log(t_intensity(w_i, b_i, g_i, t_ti))


"""
Compute spatial/temporal/spatiotemporal intensities

tn_ti: [batch, seq_len]
s_diff: [batch, seq_len, 2]
inv_var = [batch, seq_len, 2]
w_i: [batch, seq_len]
b_i: [batch, seq_len]
g_i: [batch, seq_len]

return: λ(t) [batch]
return: f(s|t) [batch] 
return: λ(s,t) [batch]
"""
def t_intensity(w_i, b_i, g_i, t_ti):
    v_i = w_i * torch.exp(-(torch.log(t_ti) - b_i) ** 2 / 2) / t_ti
    lamb_t = torch.sum(v_i, -1)
    return lamb_t

def s_intensity(w_i, b_i, t_ti, s_diff, inv_var):
    v_i = w_i * torch.exp(-b_i * t_ti)
    v_i = v_i / torch.sum(v_i, -1).unsqueeze(-1) # normalize
    g2 = torch.sum(s_diff * inv_var * s_diff, -1)
    g2 = torch.sqrt(torch.prod(inv_var, -1)) * torch.exp(-0.5*g2)/(2*np.pi)
    f_s_cond_t = torch.sum(g2 * v_i, -1)
    return f_s_cond_t

def intensity(w_i, b_i, t_ti, s_diff, inv_var):
    return t_intensity(w_i, b_i, t_ti) * s_intensity(w_i, b_i, t_ti, s_diff, inv_var)



"""
STPP model with VAE: directly modeling λ(s,t)
"""
class DeepSTPP(nn.Module):
    def __init__(self, config, device):
        super(DeepSTPP, self).__init__()
        self.config = config
        self.emb_dim = config.emb_dim
        self.hid_dim = config.hid_dim
        self.device = device
        
        # VAE for predicting spatial intensity
        self.enc = Encoder(config, device)
        
        output_dim = config.seq_len + config.num_points
        self.w_dec = Decoder(config, output_dim, softplus=True)
        self.b_dec = Decoder(config, output_dim)
        self.s_dec = Decoder(config, output_dim * 2, softplus=True)
        self.g_dec = Decoder(config, output_dim, softplus=True)
        
        # Set prior as fixed parameter attached to Module
        self.z_prior_m = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)
        
        # Background 
        self.num_points = config.num_points
        self.background = nn.Parameter(torch.rand((self.num_points, 2)), requires_grad=True)
        
        self.optimizer = self.set_optimizer(config.opt, config.lr, config.momentum)
        self.to(device)

    
    """
    st_x: [batch, seq_len, 3] (lat, lon, time)
    st_y: [batch, 1, 3]
    """
    def loss(self, st_x, st_y):
        batch = st_x.shape[0]
        background = self.background.unsqueeze(0).repeat(batch, 1, 1)
        
        s_diff = st_y[..., :2] - torch.cat((st_x[..., :2], background), 1) # s - s_i
        t_cum = torch.cumsum(st_x[..., 2], -1)
        
        tn_ti = t_cum[..., -1:] - t_cum # t_n - t_i
        tn_ti = torch.cat((tn_ti, torch.zeros(batch, self.num_points).to(self.device)), -1)
        t_ti  = tn_ti + st_y[..., 2] # t - t_i

        [qm, qv], w_i, b_i, g_i, inv_var = self(st_x)

        # KL Divergence
        kl = kl_normal(qm, qv, *self.z_prior).mean()
            
        # Calculate likelihood
        sll = torch.log(s_intensity(w_i, b_i, t_ti, s_diff, inv_var))
        tll = log_ft(t_ti, tn_ti, w_i, b_i, g_i)
        
        nelbo = kl - self.config.beta * (sll.mean() + tll.mean())

        return nelbo, sll, tll
   
    
    def forward(self, st_x):        
        # Encode history locations and times
        qm, qv = self.enc.encode(st_x) # Variational posterior
        
        # Monte Carlo
        z = sample_gaussian(qm, qv)
        w_i = self.w_dec.decode(z)
        b_i = self.b_dec.decode(z)
        s_i = self.s_dec.decode(z) + 1e-2
        g_i = self.g_dec.decode(z)
        
        s_x, s_y = torch.split(s_i, s_i.size(-1) // 2, dim=-1)
        inv_var = torch.stack((1 / s_x, 1 / s_y), -1)

        return [qm, qv], w_i, b_i, g_i, inv_var
  
    
    def set_optimizer(self, opt, lr, momentum):
        if opt == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        elif opt == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=lr)