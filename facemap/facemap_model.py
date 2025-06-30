# FaceMap Model - Omar

# %% 
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import zscore

import os
import numpy as np

from scipy.io import savemat


## --------------------------------------------------------------------------- CLASS CORE -----------------------------------------------------------
# --------- Linear --> 1D CONV --> ReLU --> Linear --ReLU ===== Latents

class Core(nn.Module):
    def __init__(self, n_in=55, n_kp=None, n_filt=10, kernel_size=201, n_layers=1, n_med=50, n_latents=256, identity=False, relu_wavelets=True, relu_latents=True):
        super().__init__()
        self.n_in = n_in
        self.n_kp = n_in if n_kp is None or identity else n_kp
        self.n_filt = (n_filt // 2) * 2  # must be even for initialization
        self.relu_latents = relu_latents
        self.relu_wavelets = relu_wavelets
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.n_latents = n_latents
        self.features = nn.Sequential()

        # Reduces Keypoints Dim (n_in) to Hidden Dim (n_kp) in first layer ("linear0")
        if identity:
            self.features.add_module("linear0", nn.Identity(self.n_in))
        else:
            self.features.add_module(
                "linear0",
                nn.Sequential(
                    nn.Linear(self.n_in, self.n_kp),
                ),
            )
        
        def gabor_wavelet(sigma, f, ph, n_pts=201, is_torch=False):
            x = np.linspace(0, 2 * np.pi, n_pts + 1)[:-1].astype("float32")
            cos = np.cos
            sin = np.sin
            exp = np.exp
            xc = x - x.mean()
            cosine = cos(ph + f * xc)
            gaussian = exp(-(xc**2) / (2 * sigma**2))
            G = gaussian * cosine
            G /= (G**2).sum() ** 0.5
            return G

        # Initalize Gabor Wavelets - 1D CONV
        f = np.geomspace(1, 10, self.n_filt // 2).astype("float32")
        gw0 = gabor_wavelet(1, f[:, np.newaxis], 0, n_pts=kernel_size) # First Gabor Wavelet filter into 0 degrees orientation
        gw1 = gabor_wavelet(1, f[:, np.newaxis], np.pi / 2, n_pts=kernel_size) # Second Gabor Wavelet filter into 90 degrees orientation 
        wav_init = np.vstack((gw0, gw1))

        # compute n_filt wavelet features of each one => n_filt * n_kp features
        self.features.add_module(
            "wavelet0",
            nn.Conv1d(
                1,
                self.n_filt,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
        )

        self.features[-1].weight.data = torch.from_numpy(wav_init).unsqueeze(1) # Sets Initial Weights of Wavelet Filters

        # Adds Linear layers (n_layers) to features
        for n in range(1, n_layers):
            n_in = self.n_kp * self.n_filt if n == 1 else n_med
            self.features.add_module(f"linear{n}", nn.Sequential(nn.Linear(n_in, n_med)))

        # Adds the final layer (n_med to n_latents)
        n_med = n_med if n_layers > 1 else self.n_filt * self.n_kp
        self.features.add_module("latent", nn.Sequential(nn.Linear(n_med, n_latents)))



    def wavelets(self, x):
        """compute wavelets of keypoints through linear + conv1d + relu layer"""
        # x is (n_batches, time, features)
        out = self.features[0](x.reshape(-1, x.shape[-1]))
        out = out.reshape(x.shape[0], x.shape[1], -1).transpose(2, 1)
        # out is now (n_batches, n_kp, time)
        out = out.reshape(-1, out.shape[-1]).unsqueeze(1)
        # out is now (n_batches * n_kp, 1, time)
        out = self.features[1](out)
        # out is now (n_batches * n_kp, n_filt, time)
        out = out.reshape(-1, self.n_kp * self.n_filt, out.shape[-1]).transpose(2, 1)
        out = out.reshape(-1, self.n_kp * self.n_filt)

        if self.relu_wavelets:
            out = F.relu(out)
        
        # if n_layers > 1, go through more linear layers
        for n in range(1, self.n_layers):
            out = self.features[n + 1](out)
            out = F.relu(out)
        return out

    def forward(self, x=None, wavelets=None):
        """x is (n_batches, time, features)
        sample_inds is (sub_time) over batches
        """
        if wavelets is None:
            wavelets = self.wavelets(x)
        wavelets = wavelets.reshape(-1, wavelets.shape[-1])

        # latent layer
        latents = self.features[-1](wavelets)
        latents = latents.reshape(x.shape[0], -1, latents.shape[-1])
        if self.relu_latents:
            latents = F.relu(latents)
        latents = latents.reshape(-1, latents.shape[-1])
        return latents


## --------------------------------------------------------------------------- CLASS READOUT -----------------------------------------------------------
# --------- Latents --> Session-specific Linear Layer === Neural PCs

class Readout(nn.Module):
    """Linear Layers from Latents (Core Class) ---> Neural PCs with session-specific output dimensions."""
    def __init__(self, out_list, sessions_list, n_latents=256, n_layers=1, n_med=128):
        super().__init__()
        self.sessions_list = sessions_list
        self.n_latents = n_latents
        self.n_layers = n_layers
        self.n_med = n_med
        self.out_list = out_list
        self.features = nn.ModuleDict()
        
        # Define separate layers for each session, with dynamic n_out from out_list
        for session_id in sessions_list:
            session_layers = nn.Sequential()
            for j in range(n_layers):
                n_in = self.n_latents if j == 0 else self.n_med
                n_outc = out_list[session_id - 1] if j == n_layers - 1 else self.n_med
                session_layers.add_module(f"linear{j}", nn.Linear(n_in, n_outc))
                if self.n_layers > 1 and j < self.n_layers - 1:
                    session_layers.add_module(f"relu{j}", nn.ReLU())
            self.features[f"session_{session_id}"] = session_layers
            
    def forward(self, latents, session_id):        
        session_key = f"session_{session_id}"
        if session_key in self.features:
            output = self.features[session_key](latents)
            # print(f"Session ID: {session_id}, Expected Output Shape: {self.out_list[session_id - 1]}, Actual Output Shape: {output.shape}")
            return output
        else:
            print(f"Unexpected Session ID: {session_id}. Returning latents unchanged.")
            return latents

        

class KeypointsNetwork(nn.Module):
    """Keypoints to neural PCs / neural activity model"""
    def __init__(self, n_in=55, n_kp=None, n_filt=10, kernel_size=201, n_core_layers=2, n_latents=256, out_list=None, sessions_list=None, n_out_layers=1, n_med=50, identity=False, relu_wavelets=True, relu_latents=True):
        super().__init__()
        self.core = Core(
            n_in=n_in,
            n_kp=n_kp,
            n_filt=n_filt,
            kernel_size=kernel_size,
            n_layers=n_core_layers,
            n_med=n_med,
            n_latents=n_latents,
            identity=identity,
            relu_wavelets=relu_wavelets,
            relu_latents=relu_latents,
        )
        
        # Ensure out_list is provided and matches the expected number of sessions
        if out_list is None or len(out_list) == 0:
            raise ValueError("out_list must be provided and cannot be empty.")
        
        self.readout = Readout(
            out_list=out_list, 
            sessions_list = sessions_list,
            n_latents=n_latents,
            n_layers=n_out_layers,
            n_med=n_med
        )

        # Store out_list for session ID validation
        self.valid_session_ids = sessions_list

    def forward(self, x, sample_inds=None):
        
        # Run through the Core to get latents
        latents = self.core(x)

        # Sample latents if indices are provided
        if sample_inds is not None:
            latents = latents[sample_inds]
        
        latents = latents.reshape(x.shape[0], -1, latents.shape[-1])  # Reshape latents for output


        return latents
    
