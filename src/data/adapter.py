import numpy as np
import torch

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader


class DataAdapter:
    """
    seq: (N, 3), [lat, lon, time]
    """
    def __init__(self, st_data, lookback=10, lookahead=1, split=None):
        self.st_data = st_data
        
        # t -> delta_t
        st_data[:, -1] = np.diff(np.append(0, st_data[:, -1]))
        
        # Default split is train:val:test = 8:1:1
        if split is None:
            split = [8, 1, 1]
        split = split / np.sum(split)

        # Min-max scale spatiotemporal data
        self.st_scaler = MinMaxScaler()
        self.st_scaler.fit(st_data)
        st_data = self.st_scaler.transform(st_data)
        
        # Breaking sequence to training data: [1 ~ N] -> [1 ~ lookback][lookback+1],
        # [2 ~ lookback+1][lookback+2]...
        
        length = len(st_data) - lookback - lookahead
        st_input = np.zeros((length, lookback, 3))
        st_label = np.zeros((length, lookahead, 3))
        
        train_size = int(split[0] * length)
        test_size = int(split[2] * length)

        for i in range(length):
            st_input[i] = st_data[i:i + lookback]
            st_label[i] = st_data[i + lookback:i + lookback + lookahead]

        self.train = TensorDataset(torch.Tensor(st_input[:train_size]),
                                   torch.Tensor(st_label[:train_size]))

        self.val = TensorDataset(torch.Tensor(st_input[train_size:-test_size]),
                                 torch.Tensor(st_label[train_size:-test_size]))

        self.test = TensorDataset(torch.Tensor(st_input[-test_size:]),
                                  torch.Tensor(st_label[-test_size:]))