import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm


LRELU_SLOPE = 0.1


def _analytic_signal(x):
    length = x.shape[-1]
    spectrum = torch.fft.fft(x, dim=-1)
    multiplier = torch.zeros(length, device=x.device, dtype=x.dtype)
    if length % 2 == 0:
        multiplier[0] = 1
        multiplier[length // 2] = 1
        multiplier[1 : length // 2] = 2
    else:
        multiplier[0] = 1
        multiplier[1 : (length + 1) // 2] = 2
    return torch.fft.ifft(spectrum * multiplier.view(1, 1, length), dim=-1)


class FixedEnvelope(nn.Module):
    """Fixed MED envelope transform with batch-safe torch FFT operations."""

    def __init__(self, max_freq, sample_rate=16000):
        super().__init__()
        self.max_freq = float(max_freq)
        self.sample_rate = sample_rate

    def forward(self, x):
        if self.max_freq == 0:
            return x
        if self.max_freq == -1:
            return -torch.abs(_analytic_signal(-x))
        if self.max_freq == 1:
            return torch.abs(_analytic_signal(x))

        length = x.shape[-1]
        spectrum = torch.fft.fft(x, dim=-1)
        freqs = torch.fft.fftfreq(length, d=1.0 / self.sample_rate).to(x.device)
        mask = (freqs.abs() <= self.max_freq).to(dtype=x.dtype).view(1, 1, length)
        lowpassed = torch.fft.ifft(spectrum * mask, dim=-1).real
        return torch.abs(_analytic_signal(lowpassed))


class DiscriminatorEForASD(nn.Module):
    """BemaGANv2 MED branch repurposed as a feature-map extractor."""

    def __init__(self, max_freq, sample_rate=16000, use_spectral_norm=False):
        super().__init__()
        self.envelope = FixedEnvelope(max_freq=max_freq, sample_rate=sample_rate)
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
                norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = self.envelope(x)
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MEDFeatureExtractor(nn.Module):
    """Fixed-cutoff MED-ASD feature extractor for near/far/diff waveforms."""

    def __init__(self, cutoff_list=(-1, 0, 1, 300, 500), sample_rate=16000):
        super().__init__()
        self.cutoff_list = tuple(cutoff_list)
        self.branches = nn.ModuleList(
            [
                DiscriminatorEForASD(max_freq=cutoff, sample_rate=sample_rate)
                for cutoff in self.cutoff_list
            ]
        )

    @staticmethod
    def pool_fmaps(fmaps):
        pooled = []
        for feat in fmaps:
            pooled.append(feat.mean(dim=-1))
            pooled.append(feat.std(dim=-1, unbiased=False))
        return torch.cat(pooled, dim=1)

    def forward_one_channel(self, x):
        embeddings = []
        for branch in self.branches:
            _, fmaps = branch(x)
            embeddings.append(self.pool_fmaps(fmaps))
        return torch.cat(embeddings, dim=1)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"MEDFeatureExtractor expects [B, C, T], got {tuple(x.shape)}")
        if x.shape[1] == 1:
            near = x[:, 0:1, :]
            far = near
        else:
            near = x[:, 0:1, :]
            far = x[:, 1:2, :]
        diff = near - far

        z_near = self.forward_one_channel(near)
        z_far = self.forward_one_channel(far)
        z_diff = self.forward_one_channel(diff)
        return torch.cat([z_near, z_far, z_diff], dim=1)
