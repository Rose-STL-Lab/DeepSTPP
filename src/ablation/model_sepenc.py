import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from tqdm.auto import tqdm

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
            *[nn.Linear(config.hid_dim, config.hid_dim),
            nn.ELU()] * (config.decoder_n_layer - 1),
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
def ll_no_events(w_i, b_i, tn_ti, t_ti):
    return torch.sum(w_i / b_i * (torch.exp(-b_i * t_ti) - torch.exp(-b_i * tn_ti)), -1)


def log_ft(t_ti, tn_ti, w_i, b_i):
    return ll_no_events(w_i, b_i, tn_ti, t_ti) + torch.log(t_intensity(w_i, b_i, t_ti))

"""
Compute spatial/temporal/spatiotemporal intensities

tn_ti: [batch, seq_len]
s_diff: [batch, seq_len, 2]
inv_var = [batch, seq_len, 2]
w_i: [batch, seq_len]
b_i: [batch, seq_len]

return: λ(t) [batch]
return: f(s|t) [batch] 
return: λ(s,t) [batch]
"""
def t_intensity(w_i, b_i, t_ti):
    v_i = w_i * torch.exp(-b_i * t_ti)
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
        self.w_enc = Encoder(config, device)
        self.b_enc = Encoder(config, device)
        self.s_enc = Encoder(config, device)
        
        output_dim = config.seq_len + config.num_points
        self.w_dec = Decoder(config, output_dim, softplus=True)
        self.b_dec = Decoder(config, output_dim)
        self.s_dec = Decoder(config, output_dim * 2, softplus=True)
        
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

        [qm_w, qv_w, qm_b, qv_b, qm_s, qv_s], w_i, b_i, inv_var = self(st_x)
            
        # Calculate likelihood
        sll = torch.log(s_intensity(w_i, b_i, t_ti, s_diff, inv_var))
        tll = log_ft(t_ti, tn_ti, w_i, b_i)
        
        # KL Divergence
        if self.config.sample:
            kl = kl_normal(qm_w, qv_w, *self.z_prior).mean() + \
            kl_normal(qm_b, qv_b, *self.z_prior).mean() + \
            kl_normal(qm_s, qv_s, *self.z_prior).mean() 
            nelbo = kl - self.config.beta * (sll.mean() + tll.mean())
        else:
            nelbo = - (sll.mean() + tll.mean())

        return nelbo, sll, tll
   
    
    def forward(self, st_x):
        if self.config.sample:
            # Encode history locations and times
            qm_w, qv_w = self.w_enc.encode(st_x) # Variational posterior
            qm_b, qv_b = self.b_enc.encode(st_x)
            qm_s, qv_s = self.s_enc.encode(st_x)
            # Monte Carlo
            z_w = sample_gaussian(qm_w, qv_w)
            z_b = sample_gaussian(qm_b, qv_b)
            z_s = sample_gaussian(qm_s, qv_s)
        else:
            qm_w, qv_w, qm_b, qv_b, qm_s, qv_s = None, None, None, None, None, None
            z_w, _ = self.w_enc.encode(st_x)
            z_b, _ = self.b_enc.encode(st_x)
            z_s, _ = self.s_enc.encode(st_x)

        w_i = self.w_dec.decode(z_w)
        if self.config.constrain_b is 'tanh':
            b_i = torch.tanh(self.b_dec.decode(z_b)) * self.config.b_max
        elif self.config.constrain_b is 'sigmoid':
            b_i = torch.sigmoid(self.b_dec.decode(z_b)) * self.config.b_max
        elif self.config.constrain_b is 'softplus':
            b_i = torch.nn.functional.softplus(self.b_dec.decode(z_b))
        elif self.config.constrain_b is 'clamp':
            b_i = torch.clamp(self.b_dec.decode(z_b), -self.config.b_max, self.config.b_max)
        else:
            b_i = self.b_dec.decode(z_b)
                    
        s_i = self.s_dec.decode(z_s) + self.config.s_min
        
        s_x, s_y = torch.split(s_i, s_i.size(-1) // 2, dim=-1)
        inv_var = torch.stack((1 / s_x, 1 / s_y), -1)

        return [qm_w, qv_w, qm_b, qv_b, qm_s, qv_s], w_i, b_i, inv_var
  
    
    def set_optimizer(self, opt, lr, momentum):
        if opt == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        elif opt == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=lr)


"""
Calculate the uniformly samplded spatiotemporal intensity with a given
number of spatiotemporal steps  
"""
def calc_lamb(model, test_loader, config, device, scales=np.ones(3), biases=np.zeros(3),
              t_nstep=201, x_nstep=101, y_nstep=101, total_time=None, round_time=True,
              xmax=None, xmin=None, ymax=None, ymin=None):
    
    # Aggregate data
    st_xs = []
    st_ys = []
    st_x_cums = []
    st_y_cums = []
    for data in test_loader:
        st_x, st_y, st_x_cum, st_y_cum, (idx, _) = data
        mask = idx == 0 # Get the first sequence only
        st_xs.append(st_x[mask])
        st_ys.append(st_y[mask])
        st_x_cums.append(st_x_cum[mask])
        st_y_cums.append(st_y_cum[mask])

        if not torch.any(mask):
            break
        
    # Stack the first sequence
    st_x = torch.cat(st_xs, 0).cpu()
    st_y = torch.cat(st_ys, 0).cpu()
    st_x_cum = torch.cat(st_x_cums, 0).cpu()
    st_y_cum = torch.cat(st_y_cums, 0).cpu()
    if total_time is None:
        total_time = st_y_cum[-1, -1, -1]

    print(f'Intensity time range : {total_time}')
    lambs = []
    
    # Discretize space
    if xmax is None:
        xmax = 1.0
        xmin = 0.0
        ymax = 1.0
        ymin = 0.0
    else:
        xmax = (xmax - biases[0]) / scales[0]
        xmin = (xmin - biases[0]) / scales[0]
        ymax = (ymax - biases[1]) / scales[1]
        ymin = (ymin - biases[1]) / scales[1]

    x_step = (xmax - xmin) / (x_nstep - 1)
    y_step = (ymax - ymin) / (y_nstep - 1)
    x_range = torch.arange(xmin, xmax + 1e-10, x_step)
    y_range = torch.arange(ymin, ymax + 1e-10, y_step) 
    s_grids = torch.stack(torch.meshgrid(x_range, y_range), dim=-1).view(-1, 2)
    
    # Discretize time
    t_start = st_x_cum[0, -1, -1].item()
    t_step = (total_time - t_start) / (t_nstep - 1)
    if round_time:
        t_range = torch.arange(round(t_start)+1, round(total_time), 1.0)
    else:
        t_range = torch.arange(t_start, total_time, t_step)
        
    # Calculate intensity
    background = model.background.unsqueeze(0).cpu().detach()

    # Sample model parameters
    _, w_i, b_i, inv_var = model(st_x.to(device))
    w_i  = w_i.cpu().detach()
    b_i  = b_i.cpu().detach()
    inv_var = inv_var.cpu().detach()
    
    # Convert to history
    his_st     = torch.vstack((st_x[0], st_y.squeeze())).numpy()
    his_st_cum = torch.vstack((st_x_cum[0], st_y_cum.squeeze())).numpy()

    for t in tqdm(t_range):
        i = sum(st_x_cum[:, -1, -1] <= t) - 1 # index of corresponding history events

        st_x_ = st_x[i:i+1]
        w_i_ = w_i[i:i+1]
        b_i_ = b_i[i:i+1]
        inv_var_ = inv_var[i:i+1]

        t_ = t - st_x_cum[i:i+1, -1, -1] # time since lastest event
        t_ = (t_ - biases[-1]) / scales[-1]

        # Calculate temporal intensity
        t_cum = torch.cumsum(st_x_[..., -1], -1)
        tn_ti = t_cum[..., -1:] - t_cum # t_n - t_i
        tn_ti = torch.cat((tn_ti, torch.zeros(1, config.num_points)), -1)
        t_ti  = tn_ti + t_

        lamb_t = t_intensity(w_i_, b_i_, t_ti) / np.prod(scales)

        # Calculate spatial intensity
        N = len(s_grids) # number of grid points

        s_x_ = torch.cat((st_x_[..., :-1], background), 1).repeat(N, 1, 1)
        s_diff = s_grids.unsqueeze(1) - s_x_
        lamb_s = s_intensity(w_i_.repeat(N, 1), b_i_.repeat(N, 1), t_ti.repeat(N, 1), 
                             s_diff, inv_var_.repeat(N, 1, 1))
        #print(lamb_t)
        #print(torch.max(lamb_s))
        #print('-----------')

        lamb = (lamb_s * lamb_t).view(x_nstep, y_nstep)
        lambs.append(lamb.numpy())

    x_range = x_range.numpy() * scales[0] + biases[0]
    y_range = y_range.numpy() * scales[1] + biases[1]
    t_range = t_range.numpy()

    return lambs, x_range, y_range, t_range, his_st_cum[:, :2], his_st_cum[:, 2]
