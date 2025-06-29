'''
Based on 'waveform.py' from sirentv repo (originally from larnd sim), but making following changes:
    - Scintillation Modlel: Instead of computing pdf from cdf, we sample from 
      exponential distribution of arrival times for each tick
'''

import math
from functools import partial
from typing import Callable, Dict, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os

def print_grad(name):
    def hook(grad):
        print(f"Gradient for {name}: {grad}")
    return hook

T = TypeVar("T")
class Config(Dict[str, T]):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getattr__(self, name: str) -> T:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: T) -> None:
        if isinstance(value, str):
            value = eval(value, {}, {"uniform": np.random.uniform})
        self[name] = value

class BatchedLightSimulation(nn.Module):
    def __init__(self, cfg: str = os.path.join(os.path.dirname(__file__), "../templates/waveform_sim.yaml"), verbose: bool =False):
        super().__init__()

        if isinstance(cfg, str):
            cfg_txt = open(cfg,'r').readlines()
            print("BatchedLightSimulation Config:\n\t%s" % '\t'.join(cfg_txt))
            cfg = yaml.safe_load(''.join(cfg_txt))
            cfg = Config(**cfg)

        self.cfg = cfg

        # Load in and transform parameters
        def logit(x):
            return np.log(x / (1 - x))

        def create_parameter(value, nominal):
            return nn.Parameter(torch.tensor(value / nominal, dtype=torch.float32))

        # Calculate and store nominal values
        self.nominal_values = {
            'singlet_fraction_logit': logit(cfg.NOMINAL_SINGLET_FRACTION),
            'log_tau_s': np.log10(cfg.NOMINAL_TAU_S),
            'log_tau_t': np.log10(cfg.NOMINAL_TAU_T),
            'log_light_oscillation_period': np.log10(cfg.NOMINAL_LIGHT_OSCILLATION_PERIOD),
            'log_light_response_time': np.log10(cfg.NOMINAL_LIGHT_RESPONSE_TIME),
            'light_gain': cfg.NOMINAL_LIGHT_GAIN
        }

        # Calculate current values
        current_values = {
            'singlet_fraction_logit': logit(cfg.SINGLET_FRACTION),
            'log_tau_s': np.log10(cfg.TAU_S),
            'log_tau_t': np.log10(cfg.TAU_T),
            'log_light_oscillation_period': np.log10(cfg.LIGHT_OSCILLATION_PERIOD),
            'log_light_response_time': np.log10(cfg.LIGHT_RESPONSE_TIME),
            'light_gain': cfg.LIGHT_GAIN
        }

        # Create parameters
        for name in self.nominal_values.keys():
            setattr(self, name, create_parameter(current_values[name], self.nominal_values[name]))
            setattr(self, 'nominal_' + name, self.nominal_values[name])

        # Constants
        self.light_tick_size = cfg.LIGHT_TICK_SIZE
        self.light_window = cfg.LIGHT_WINDOW

        self.conv_ticks = math.ceil(
            (self.light_window[1] - self.light_window[0]) / self.light_tick_size
        )

        self.time_ticks = torch.arange(self.conv_ticks)

        self.k = 100

        if verbose:
            self.register_grad_hook()

    def to(self, device):
        self.time_ticks = self.time_ticks.to(device)
        for param in ['log_light_oscillation_period', 'log_light_response_time', 'light_gain']:
            if not isinstance(getattr(self, param), nn.Parameter):
                setattr(self, param, getattr(self, param).to(device))
        super().to(device)
        return self
    
    @property
    def device(self):
        return next(self.parameters()).device

    def register_grad_hook(self):
        for name, p in self.named_parameters():
            p.register_hook(print_grad(name))

    def scintillation_model_sampled(self, timing_dist: torch.Tensor) -> torch.Tensor:
        '''
        Parameters:
            - timing_dist: Tensor of shape (ndet, nticks) representing photon counts per time tick for each detector.
                           We assume only time ticks 2560 to 2570 are non-zero.
        Returns:
            - Tensor of shape (ndet, nticks) storing updated photon hit counts after applying scintillation delays.
        '''
        # Debugging things
        info = dict()
        info['num_singlets'] = 0
        info['num_triplets'] = 0
        ##################
        
        device = timing_dist.device
        if timing_dist.ndim < 3:
            timing_dist = timing_dist.unsqueeze(0)
        nbatch, ndet, nticks = timing_dist.shape
    
        # Define time ticks
        time_ticks = torch.arange(nticks, device=device)
        bin_width = self.light_tick_size  # Size of each time bin in microseconds

        all_emission_delays = [] # in units of microseconds
        delayed_counts = torch.zeros_like(timing_dist) # in units of time ticks
        
        for batch in range(nbatch):
            for det in range(ndet):
                for tick in range(2550, 2600):  # Inclusive range
                    n_photons = int(timing_dist[batch, det, tick].item())
                    if n_photons == 0:
                        continue
        
                    # Sample singlet vs triplet for each photon
                    singlet_fraction = self.cfg.NOMINAL_SINGLET_FRACTION
                    is_singlet = torch.rand((n_photons,), device=device) < singlet_fraction
        
                    # Sample emission delays
                    tau_s = self.cfg.TAU_S
                    tau_t = self.cfg.TAU_T

                    # print(f"tau_s & tau_t: {tau_s, tau_t})")
                    num_singlets = is_singlet.sum().item()
                    singlet_delays = torch.distributions.Exponential(1.0 / tau_s).sample((num_singlets,)).to(device)

                    num_triplets = (~is_singlet).sum().item()
                    triplet_delays = torch.distributions.Exponential(1.0 / tau_t).sample((num_triplets,)).to(device)

                    info['num_singlets'] += num_singlets
                    info['num_triplets'] += num_triplets

                    emission_delays = torch.cat([singlet_delays, triplet_delays])
                    all_emission_delays.append(emission_delays)
        
                    # Map delays to time bins
                    base_time = tick * self.light_tick_size  # microseconds
                    photon_times = base_time + emission_delays  # absolute photon times in microseconds
        
                    bin_centers = time_ticks * self.light_tick_size
                    bin_edges = torch.cat([
                        bin_centers[:1] - bin_width / 2,  # left edge of first bin
                        bin_centers + bin_width / 2      # right edges
                    ])
        
                    bin_indices = torch.bucketize(photon_times, bin_edges) - 1  # Adjust for 0-based indexing
        
                    # Count photons in each bin
                    valid_mask = (bin_indices >= 0) & (bin_indices < nticks)
                    bin_indices = bin_indices[valid_mask]
                    counts = torch.bincount(bin_indices, minlength=nticks)
        
                    delayed_counts[batch, det] += counts
    
        return all_emission_delays, delayed_counts, info

    def tpb_delay(self, timing_dist: torch.Tensor, tau=0.002):
        '''
        Parameters:
            - timing_dist: Tensor of shape (nbatch, ndet, nticks) representing photon counts per time tick for each detector.
            - tau: lifetime of tpb re-emission spectrum
        Returns:
            - Tensor of shape (ndet, nticks) storing updated photon hit counts after applying scintillation delays.
        '''
        device = timing_dist.device
        if timing_dist.ndim < 3:
            timing_dist = timing_dist.unsqueeze(0)
        nbatch, ndet, nticks = timing_dist.shape
    
        # Define time ticks
        time_ticks = torch.arange(nticks, device=device)
        bin_width = self.light_tick_size  # Size of each time bin in microseconds

        all_emission_delays = []

        delayed_counts = torch.zeros_like(timing_dist) # in units of time ticks
        
        for batch in range(nbatch):
            # get nonzero (pmt_id, tick) indices in timing_dist
            nonzero_indices = torch.nonzero(timing_dist)
            
            for i, (pmt_id, tick) in enumerate(nonzero_indices):
                n_photons = int(timing_dist[batch, det, tick].item())
                emission_delays = torch.distributions.Exponential(1.0 / tau).sample((n_photons,)).to(device)
                all_emission_delays.append(emission_delays)
    
                # Map delays to time bins
                base_time = tick * self.light_tick_size  # microseconds
                photon_times = base_time + emission_delays  # absolute photon times in microseconds
    
                bin_centers = time_ticks * self.light_tick_size
                bin_edges = torch.cat([
                    bin_centers[:1] - bin_width / 2,  # left edge of first bin
                    bin_centers + bin_width / 2      # right edges
                ])
    
                bin_indices = torch.bucketize(photon_times, bin_edges) - 1  # Adjust for 0-based indexing
    
                # Count photons in each bin
                valid_mask = (bin_indices >= 0) & (bin_indices < nticks)
                bin_indices = bin_indices[valid_mask]
                counts = torch.bincount(bin_indices, minlength=nticks)
    
                delayed_counts[batch, det] += counts
            
        return all_emission_delays, delayed_counts

    def gen_waveform(self, mode='precise', **kwargs):
        if mode == 'precise':
            return self._gen_waveform_precise(**kwargs)
        elif mode == 'gaussian':
            return self._gen_waveform_gaussian(**kwargs)
        else:
            pass
            
    def _gen_waveform_precise(self, pmt_ids: torch.Tensor, arrival_times: torch.Tensor, nphotons:int=1, n_pmts:int=128):
        '''
        Parameters:
            - pmt_ids: tensor of shape (n_photons,), integers in [0, n_pmts)
            - arrival_times: tensor of shape (n_photons,), integer tick values
            - n_pmts: number of PMTs (default 128)
        Returns:
            - waveform: tensor of shape (n_pmts, n_ticks), where each row is a histogram of arrivals over time
                        photon arrival times are offset by 2560 ticks
        '''
        assert pmt_ids.shape == arrival_times.shape

        n_ticks = 16000

        arrival_times = arrival_times + 2560 # adding an offset

        waveform = torch.zeros((n_pmts, n_ticks))
        waveform.index_put_((pmt_ids, arrival_times), torch.full_like(pmt_ids, nphotons, dtype=waveform.dtype), accumulate=True)

        return waveform

    def _gen_waveform_gaussian(self, pmt_ids: torch.Tensor, nphotons: torch.Tensor, arrival_mean:float=0.0, std:float=1.0, n_pmts:int=128):
        '''
        Parameters:
            - pmt_ids: tensor of shape (<=n_pmts,), integers in [0, n_pmts)
            - nphotons: tensor of shape (<=n_pmts,); number of photons per pmt
            - arrival_mean: mean for arrival time gaussian
            - std: gaussian std for arrival time sampling (default 1 microsecond)
            - n_pmts: number of PMTs (default 128)
        '''
        assert pmt_ids.shape == nphotons.shape

        n_ticks = 16000 # 16000 nanoseconds / 16 microseconds

        # for each pmt in pmt_ids
        # for each photon in nphotons[pmt_id]
        # sample arrival time from gaussian w std centered around 0
        waveform = torch.zeros((n_pmts, n_ticks))
        
        for i, pmt_id in enumerate(pmt_ids):
            pmt_arrival_times = torch.normal(mean=arrival_mean + 2560, std=std * 1000, size=(nphotons[i],)) # offsetting arrival mean
            arrival_bins = torch.clamp(pmt_arrival_times.round().long(), min=0, max=n_ticks - 1)
            waveform.index_put_((torch.full_like(arrival_bins, pmt_id), arrival_bins),
                torch.ones_like(arrival_bins, dtype=waveform.dtype),
                accumulate=True)

        return waveform
            
        
    def scintillation_model(self, time_tick: torch.Tensor, relax_cut: bool=True) -> torch.Tensor:
        """
        Calculates the fraction of scintillation photons emitted
        during time interval `time_tick` to `time_tick + 1`

        Args:
            time_tick (torch.Tensor): time tick relative to t0
            relax_cut (bool): whether to apply the relaxing cut for differentiability

        Returns:
            torch.Tensor: fraction of scintillation photons
        """

        singlet_fraction = torch.sigmoid(self.singlet_fraction_logit * self.nominal_singlet_fraction_logit)
        tau_s = torch.pow(10, self.log_tau_s * self.nominal_log_tau_s)
        tau_t = torch.pow(10, self.log_tau_t * self.nominal_log_tau_t)
        t = time_tick * self.light_tick_size

        p1 = (
            singlet_fraction
            * torch.exp(-t / tau_s)
            * (1 - torch.exp(-self.light_tick_size / tau_s))
        )
        p3 = (
            (1 - singlet_fraction)
            * torch.exp(-t / tau_t)
            * (1 - torch.exp(-self.light_tick_size / tau_t))
        )
        
        if relax_cut:
            return (p1 + p3) / (1 + torch.exp(-self.k * t))

        return (p1 + p3) * (t >= 0).float()
    
    def sipm_response_model(self, time_tick, relax_cut=True) -> torch.Tensor:
        """
        Calculates the SiPM response from a PE at `time_tick` relative to the PE time

        Args:
            time_tick (torch.Tensor): time tick relative to t0
            relax_cut (bool): whether to apply the relaxing cut for differentiability

        Returns:
            torch.Tensor: response
        """

        t = time_tick * self.light_tick_size
        light_oscillation_period = torch.pow(10, self.log_light_oscillation_period * self.nominal_log_light_oscillation_period)
        light_response_time = torch.pow(10, self.log_light_response_time * self.nominal_log_light_response_time)

        impulse = torch.exp(-t / light_response_time) * torch.sin(
            t / light_oscillation_period
        )
        if relax_cut:
            impulse = impulse / (1 + torch.exp(-self.k * time_tick))
        else:
            impulse = impulse * (time_tick >= 0).float()

        impulse /= light_oscillation_period * light_response_time**2
        impulse *= light_oscillation_period**2 + light_response_time**2
        return impulse * self.light_tick_size

    def fft_conv(self, light_sample_inc: torch.Tensor, model: Callable) -> torch.Tensor:
        """
        Performs a Fast Fourier Transform (FFT) convolution on the input light sample.

        Args:
            light_sample_inc (torch.Tensor): Light incident on each detector.
                Shape: (ninput, ndet, ntick)
            model (callable): Function that generates the convolution kernel.

        Returns:
            torch.Tensor: Convolved light sample.
                Shape: (ninput, ndet, ntick)

        This method applies the following steps:
        1. Pads the input tensor
        2. Computes the convolution kernel using the provided model
        3. Performs FFT on both the input and the kernel
        4. Multiplies the FFTs in the frequency domain
        5. Performs inverse FFT to get the convolved result
        6. Reshapes and trims the output to match the input shape
        """
        ninput, ndet, ntick = light_sample_inc.shape

        # Pad the input tensor
        pad_size = self.conv_ticks - 1
        padded_input = F.pad(light_sample_inc, (0, pad_size))

        # Compute kernel
        scintillation_kernel = model(self.time_ticks)
        kernel_fft = torch.fft.rfft(scintillation_kernel, n=ntick + pad_size)

        # Reshape for batched FFT convolution
        padded_input = padded_input.reshape(ninput * ndet, ntick + pad_size)

        # Perform FFT convolution
        input_fft = torch.fft.rfft(padded_input)
        output_fft = input_fft * kernel_fft.unsqueeze(0)
        output = torch.fft.irfft(output_fft, n=ntick + pad_size)

        # Reshape and trim the result to match the input shape
        output = output.reshape(ninput, ndet, -1)[:, :, :ntick]

        return output
    
    def conv(self, light_sample_inc: torch.Tensor, model: Callable) -> torch.Tensor:
        """
        Performs a grouped convolution on the input light sample.

        Args:
            light_sample_inc (torch.Tensor): Light incident on each detector.
                Shape: (ninput, ndet, ntick)
            model (callable): Function that generates the convolution kernel.

        Returns:
            torch.Tensor: Convolved light sample.
                Shape: (ninput, ndet, ntick)

        This method applies the following steps:
        1. Pads the input tensor
        2. Reshapes the input for grouped convolution
        3. Computes the convolution kernel using the provided model
        4. Performs grouped convolution using torch.nn.functional.conv1d
        5. Reshapes and trims the output to match the input shape
        """
        ninput, ndet, ntick = light_sample_inc.shape

        # Pad the input tensor
        padded_input = torch.nn.functional.pad(
            light_sample_inc, (self.conv_ticks - 1, 0)
        )

        # Reshape for grouped convolution
        padded_input = padded_input.view(1, ninput * ndet, -1)

        kernel = model(self.time_ticks).flip(0)

        # Create a separate kernel for each input and detector
        kernel = (
            kernel.unsqueeze(0).expand(ninput * ndet, -1).unsqueeze(1)
        )

        # Perform the convolution
        output = torch.nn.functional.conv1d(
            padded_input, kernel, groups=ninput * ndet
        )

        # Trim the result to match the input shape
        output = output.view(ninput, ndet, -1)[:, :, :ntick]

        return output

    def downsample_waveform(self, waveform: torch.Tensor, ns_per_tick: int = 16) -> torch.Tensor:
        """
        Downsample the input waveform by summing over groups of ticks.
        This effectively compresses the waveform in the time dimension while preserving the total signal.

        Args:
            waveform (torch.Tensor): Input waveform tensor of shape (ninput, ndet, ntick), where each tick corresponds to 1 ns.
            ns_per_tick (int, optional): Number of nanoseconds per tick in the downsampled waveform. Defaults to 16.

        Returns:
            torch.Tensor: Downsampled waveform of shape (ninput, ndet, ntick_down).
        """
        ninput, ndet, ntick = waveform.shape
        ntick_down = ntick // ns_per_tick
        downsample = waveform.view(ninput, ndet, ntick_down, ns_per_tick).sum(dim=3)
        return downsample
d
    def forward(self, timing_dist: torch.Tensor):
        '''
        Parameters:
            - timing_dist: Tensor of shape (nbatch, ndet, nticks)
        Returns:
            - Tensor of shape (nbatch, ndet, nticks) after processing.
        '''
        reshaped = False
        if timing_dist.ndim == 1:
            timing_dist = timing_dist[None, None, :]
            reshaped = True
        elif timing_dist.ndim == 2:
            timing_dist = timing_dist[None, :, :]
            reshaped = True
    
        nbatch, ndet, nticks = timing_dist.shape

        emission_delays, x, info = self.scintillation_model_sampled(timing_dist) # stochastic photon emission sampling
        tpb_emission_delays, x = self.tpb_delay(x) # stochastic re-emission from tpb
        x = self.fft_conv(x, partial(self.sipm_response_model)) # convolving with SiPM response kernel
        x = self.light_gain * self.nominal_light_gain * x
        x = self.downsample_waveform(x)
    
        if reshaped:
            return x.squeeze(0).squeeze(0), info, emission_delays
        
        return x, info, emission_delays
        
class TimingDistributionSampler:
    def __init__(self, cdf: np.ndarray, output_shape: tuple):
        super().__init__()
        self.cdf = cdf
        self.shape = tuple(output_shape)

    def __call__(self, num_photon: int):
        u = torch.rand(num_photon)
        sampled_idx = torch.searchsorted(torch.tensor(self.cdf), u)

        output = torch.zeros(self.shape)
        unique_idx, counts = torch.unique(sampled_idx, return_counts=True)
        output.view(-1)[unique_idx] = counts.float()

        output = torch.nn.functional.pad(
            output,
            (2560, 16000 - output.shape[1] - 2560),
            mode="constant",
            value=0
        )
        return output
    
    def batch_sample(self, nphoton: int, nbatch: int) -> torch.Tensor:
        return torch.stack([self(nphoton) for _ in range(nbatch)])


data_path = os.path.join(os.path.dirname(__file__), "data/lightLUT_Mod0_06052024_32.1.16_time_dist_cdf.npz")
data = np.load(data_path)
mod0_sampler = TimingDistributionSampler(**data)