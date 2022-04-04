from copy import deepcopy

import numpy as np
import torch


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu") 
#device = torch.device("cpu") 

"""
Wrap data of a spatiotemporal point process 
"""
class SlidingWindowWrapper(torch.utils.data.Dataset):
    
    """
    Take a batch of sequences, applying sliding window to each of it to create a 
    fixed length dataset.
    
    seqs: list of [seqlen, 3] np.array
    
    self.st_X: torch.tensor, [N, lookback]
    self.st_Y: torch.tensor, [N, lookahead]
    """
    def __init__(self, seqs, lookback=20, lookahead=1, normalized=False, roll=True, min=None, max=None):
        self.seqs_cum = seqs
        self.seqs = deepcopy(self.seqs_cum)
        for i, seq in enumerate(self.seqs):
            self.seqs[i][:, 0] = np.diff(seq[:, 0], axis=0, prepend=0)
            
        if roll:
            self.seqs_cum = [np.roll(seq, -1, -1) for seq in self.seqs_cum]
            self.seqs     = [np.roll(seq, -1, -1) for seq in self.seqs]
        
        #print(self.seqs_cum[0][:10])
        #print(self.seqs[0][:10])
        #print(len(self.seqs[0]))
        #print(len(self.seqs_cum[0]))
        
        self.st_X = []
        self.st_Y = []
        self.st_X_cum = []
        self.st_Y_cum = []
        self.indices = []
        
        # Create normalizer 
        if normalized and (min is None or max is None):
            temp = np.vstack(self.seqs)
            self.min = torch.tensor(np.min(temp, 0)).float().to(device)
            self.max = torch.tensor(np.max(temp, 0)).float().to(device)
        elif normalized:
            self.min = min
            self.max = max
        
        for seq_i, (seq, seq_cum) in enumerate(zip(self.seqs, self.seqs_cum)):     
            for i in range(lookback, len(seq) + 1 - lookahead):
                self.st_X_cum.append(seq_cum[i - lookback : i])
                self.st_Y_cum.append(seq_cum[i : i + lookahead])
                
                self.st_X.append(seq[i - lookback : i])
                self.st_Y.append(seq[i : i + lookahead])
                
                self.indices.append((seq_i, i)) # Get the location in original sequence
                
        self.st_X = torch.tensor(np.stack(self.st_X)).float().to(device)
        self.st_Y = torch.tensor(np.stack(self.st_Y)).float().to(device)
        
        self.st_X_cum = torch.tensor(np.stack(self.st_X_cum)).float().to(device)
        self.st_Y_cum = torch.tensor(np.stack(self.st_Y_cum)).float().to(device)
        
        if normalized:
            def scale(st):
                return (st - self.min) / (self.max - self.min)
            
            self.st_X = scale(self.st_X)
            self.st_Y = scale(self.st_Y)
        

    
    def __len__(self):
        return len(self.st_X)

    
    """
    Return
    - normalized sequence diff
    - un-normalized original sequence
    """
    def __getitem__(self, idx):
        return self.st_X[idx], self.st_Y[idx], self.st_X_cum[idx], self.st_Y_cum[idx], self.indices[idx]
        