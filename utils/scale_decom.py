import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class ts_decom(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size=5, alpha=0.1, beta=0.1, type='avgpool', down_window=2, seq_len=128):
        super(ts_decom, self).__init__()
        self.kernel_size = kernel_size
        self.alpha = alpha
        self.beta = beta
        self.type = type
        self.seq_len = seq_len

        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

        if type == 'avgpool':
            self.down_sampling = torch.nn.AvgPool1d(kernel_size=down_window)
            self.up_sampling = F.interpolate
        else:
            self.down_sampling = nn.Sequential(
                torch.nn.Linear(seq_len, seq_len // down_window),
                nn.GELU(),
                torch.nn.Linear(seq_len // down_window, seq_len // down_window)
            )
            self.up_sampling = nn.Sequential(
                torch.nn.Linear(seq_len // down_window, seq_len),
                nn.GELU(),
                torch.nn.Linear(seq_len, seq_len)
            )


    def moving_avg(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x):
        down_x = self.down_sampling(x)

        moving_mean = self.moving_avg(down_x)
        long_term_dep = down_x - moving_mean

        if self.type == 'avgpool':
            short_term_x = self.up_sampling(moving_mean, size=self.seq_len, mode='linear', align_corners=False)
            long_term_x = self.up_sampling(long_term_dep, size=self.seq_len, mode='linear', align_corners=False)
        else:
            short_term_x = self.up_sampling(moving_mean)
            long_term_x = self.up_sampling(long_term_dep)

        return short_term_x, long_term_x

class scale_decom(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size=5, alpha=0.1, beta=0.1, down_windows=None, type='avgpool', seq_len=128):
        super(scale_decom, self).__init__()
        self.down_windows = down_windows

        self.ts_decom_layers = nn.ModuleList([
                nn.Sequential(
                    ts_decom(kernel_size=kernel_size, alpha=alpha, beta=beta, down_window=down_window, type=type, seq_len=seq_len)
                )
                for down_window in self.down_windows
        ])

    def forward(self, x):
        short_term_terms = []
        long_term_terms = []
        for i in range(len(self.down_windows)):
            short_term_x, long_term_x = self.ts_decom_layers[i](x)
            short_term_terms.append(short_term_x)
            long_term_terms.append(long_term_x)

        mean_short_term_x = torch.mean(torch.stack(short_term_terms), dim=0)
        mean_long_term_x = torch.mean(torch.stack(long_term_terms), dim=0)

        return mean_short_term_x, mean_long_term_x