import math

import torch
import torch.nn as nn

class TimeDomainBranch(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super(TimeDomainBranch, self).__init__()
        
        self.stem1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.stem2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.stem3 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        B, D = x.shape
        x = x.view(B, 1, D)  # (batch_size, 1, input_dim)
        
        x = self.stem1(x)  # (batch_size, out_channels, input_dim)
        x = self.stem2(x)  # (batch_size, out_channels, input_dim)
        x = self.stem3(x)  # (batch_size, out_channels, input_dim)
        
        x = x.mean(dim=2)  # Global Average Pooling: (batch_size, out_channels)
        return x


class FrequencyDomainBranch(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, sampling_rate=125):
        super().__init__()
        self.win_length = int(sampling_rate * 2)  # 2 seconds window
        self.hop_length = int(sampling_rate * 1)  # 1 second hop
        self.n_fft = 2 ** math.ceil(math.log2(self.win_length))  # FFT size

        self.register_buffer(
            "window",
            torch.hann_window(self.win_length)
        )

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x : (B, D)
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True
        )
        
        # (B, F, T)
        spec = torch.abs(spec)

        # (B,1,F,T)
        spec = spec.unsqueeze(1)
        out = self.stem(spec)
        
        # Global Average Pooling over frequency and time dimensions
        out = out.mean(dim=[2, 3])  # (B, out_channels)
        
        return out
    
    
class FusionStem(nn.Module):
    def __init__(self, in_channels=128, out_channels=64):
        super(FusionStem, self).__init__()
        
        self.stem = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.stem(x)
    
    
class RegressionHead(nn.Module):
    def __init__(self, in_channels=64, out_channels=1):
        super(RegressionHead, self).__init__()
        
        self.head = nn.Sequential(
            nn.Linear(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.head(x)
        

class Baseline2Branch(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, sampling_rate=125):
        super(Baseline2Branch, self).__init__()
        
        self.time_branch = TimeDomainBranch(in_channels=in_channels, out_channels=out_channels)
        self.freq_branch = FrequencyDomainBranch(
            in_channels=in_channels, 
            out_channels=out_channels,
            sampling_rate=sampling_rate,
        )
        
        self.fusion_stem = FusionStem(in_channels=2*out_channels, out_channels=out_channels)
        self.regression_head = RegressionHead(in_channels=out_channels, out_channels=1)
    
    def forward(self, x):
        time_features = self.time_branch(x)  # (B, out_channels)
        freq_features = self.freq_branch(x)   # (B, out_channels)
        
        # Concatenate features from both branches
        fused_features = torch.cat((time_features, freq_features), dim=1)  # (B, 2*out_channels)
        
        # Fusion stem
        fused_output = self.fusion_stem(fused_features)  # (B, out_channels)
        
        # Regression head
        output = self.regression_head(fused_output)  # (B, 1)
        
        return output
