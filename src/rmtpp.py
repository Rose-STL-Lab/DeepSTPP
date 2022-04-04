import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from tqdm.auto import tqdm


"""
Encode time/space record to variational posterior for location latent
"""
class Encoder(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.encoder = nn.LSTM(3, config.emb_dim, config.nlayers, batch_first=True)
        self.decoder = nn.Linear(config.emb_dim, config.z_dim)
        self.init_weights()
        self.device = device
        
    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def encode(self, x, x_mask=None):
        output, _ = self.encoder(x)
        output = self.decoder(output[:, -1, :])
        return output


"""
Log likelihood of no events happening from t_n to t
- ∫_{t_n}^t λ(t') dt' 

vh: (batch,)
w: (1,)
b: (1,)
t: (batch,)

return: scalar
"""
def log_ft(vh, w, b, t):
    return vh + w * t + b + torch.exp(vh + b) / w - torch.exp(vh + w * t + b) / w

"""
Compute temporal intensities

vh: (batch,)
w: (1,)
b: (1,)
t: (batch,)

return: λ(t) (batch,)
"""
def t_intensity(vh, w, b, t):
    return torch.exp(vh + w * t + b)


"""
Model the temporal intensity λ(t)
"""
class RMTPP(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.emb_dim = config.emb_dim
        self.hid_dim = config.hid_dim
        self.device = device
        
        # LSTM for predicting spatial intensity
        self.encoder = Encoder(config, device)
        self.decoder = nn.Linear(config.z_dim, 1)
        self.log_w = torch.nn.Parameter(torch.ones(1)) 
        self.log_b = torch.nn.Parameter(torch.ones(1)) 

        self.optimizer = self.set_optimizer(config.opt, config.lr, config.momentum)
        self.to(device)


    """
    st_x: [batch, seq_len, 3] (lat, lon, time)
    st_y: [batch, 1, 3]
    """
    def loss(self, st_x, st_y):
        w = torch.exp(self.log_w)
        b = torch.exp(self.log_b)

        batch = st_x.shape[0]
        t_cum = torch.cumsum(st_x[..., 2], -1)

        h = self.encoder.encode(st_x)
        vh = self.decoder(h).squeeze(-1)
        t = st_y[..., 2].squeeze(-1)

        # Calculate likelihood
        tll = log_ft(vh, w, b, t)
        
        return -tll.mean()
   
    
    def set_optimizer(self, opt, lr, momentum):
        if opt == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        elif opt == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=lr)


"""
Calculate the uniformly sampled temporal intensity with a given
number of spatiotemporal steps  
"""
def calc_lamb(model, test_loader, config, device, scales=np.ones(3), biases=np.zeros(3),
              t_nstep=201, total_time=None, round_time=True):
    
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
        total_time = st_y_cum[-1, -1, -1].item()

    print(f'Intensity time range : {total_time}')
    lambs = []
    
    # Discretize time
    t_start = st_x_cum[0, -1, -1].item()
    t_step = (total_time - t_start) / (t_nstep - 1)
    if round_time:
        t_range = torch.arange(round(t_start)+1, round(total_time), 1.0)
    else:
        t_range = torch.arange(t_start, total_time, t_step)
    
    # Get model parameter
    w = np.exp(model.log_w.item())
    b = np.exp(model.log_b.item())
    
    # Convert to history
    his_st     = torch.vstack((st_x[0], st_y.squeeze())).numpy()
    his_st_cum = torch.vstack((st_x_cum[0], st_y_cum.squeeze())).numpy()

    for t in tqdm(t_range):
        i = sum(st_x_cum[:, -1, -1] <= t) - 1 # index of corresponding history events

        st_x_ = st_x[i:i+1]
        h = model.encoder.encode(st_x_.to(model.device))
        vh = model.decoder(h).item()

        t_ = t - st_x_cum[i:i+1, -1, -1] # time since lastest event
        t_ = (t_ - biases[-1]) / scales[-1]
        lamb_t = t_intensity(vh, w, b, t_).item()
        
        lambs.append(lamb_t / scales[-1])

    t_range = t_range.numpy()

    return lambs, t_range, his_st_cum[:, :2], his_st_cum[:, 2]
