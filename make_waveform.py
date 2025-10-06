'''
Based on 'waveform.py' from sirentv repo (originally from larnd sim), but making following changes:
    - Scintillation Modlel: Instead of computing pdf from cdf, we sample from 
      exponential distribution of arrival times for each tick
    - TPB Wavelength Shifter Model: Stochastic sampling from re-emission timing distribution with lifetime of 2 nanoseconds
'''

import math
from functools import partial
from typing import Callable, Dict, TypeVar

import numpy as np
import time
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
    '''
    Params:
    - wf_length: waveform length in nanoseconds
    - res: simulation resolution (factor of final resolution) i.e. 10 means simulate 10 ticks per nanosecond (100 picosecond resolution before binning)
    - offset: offset waveform timing from 0, in nanoseconds
    '''
    def __init__(self, cfg: str = os.path.join(os.path.dirname(__file__), "../templates/waveform_sim.yaml"), wf_length=8000, res=10, offset=10, verbose: bool=False):
        super().__init__()

        if isinstance(cfg, str):
            cfg_txt = open(cfg,'r').readlines()
            print("BatchedLightSimulation Config:\n\t%s" % '\t'.join(cfg_txt))
            cfg = yaml.safe_load(''.join(cfg_txt))
            cfg = Config(**cfg)

        self.cfg = cfg

        self.n_ticks = wf_length * res
        self.offset = offset * res

        # Load in and transform parameters
        def logit(x):
            return np.log(x / (1 - x))

        def create_parameter(value, nominal):
            return nn.Parameter(torch.tensor(value / nominal, dtype=torch.float32))
        
        # Exact nominal values
        self.singlet_fraction = cfg.NOMINAL_SINGLET_FRACTION
        self.tau_s = cfg.NOMINAL_TAU_S
        self.tau_t = cfg.NOMINAL_TAU_T
        self.light_oscillation_period = cfg.NOMINAL_LIGHT_OSCILLATION_PERIOD
        self.light_response_time = cfg.NOMINAL_LIGHT_RESPONSE_TIME
        self.light_gain_value = cfg.NOMINAL_LIGHT_GAIN

        # Calculate and store nominal values
        self.nominal_values = {
            'singlet_fraction_logit': logit(self.singlet_fraction),
            'log_tau_s': np.log10(self.tau_s),
            'log_tau_t': np.log10(self.tau_t),
            'log_light_oscillation_period': np.log10(self.light_oscillation_period),
            'log_light_response_time': np.log10(self.light_response_time),
            'light_gain': self.light_gain_value
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
        self.downsample_factor = 10 # time ticks per new bin - simulate at 100 picosecond resolution (10 sub-ticks per nanosecond)

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
    
    def safe_logit(self, p, eps=1e-9):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))
    
    def reconfigure(self, params: dict):
        # Only update attributes if present in params, otherwise leave as-is
        if 'singlet_fraction' in params:
            self.singlet_fraction = params['singlet_fraction']
            with torch.no_grad():
                self.singlet_fraction_logit.data = torch.tensor(
                    self.safe_logit(self.singlet_fraction) / self.nominal_singlet_fraction_logit,
                    dtype=torch.float32,
                    device=self.singlet_fraction_logit.device
                )
        if 'tau_s' in params:
            self.tau_s = params['tau_s']
            with torch.no_grad():
                self.log_tau_s.data = torch.tensor(
                    np.log10(self.tau_s) / self.nominal_log_tau_s,
                    dtype=torch.float32,
                    device=self.log_tau_s.device
                )
        if 'tau_t' in params:
            self.tau_t = params['tau_t']
            with torch.no_grad():
                self.log_tau_t.data = torch.tensor(
                    np.log10(self.tau_t) / self.nominal_log_tau_t,
                    dtype=torch.float32,
                    device=self.log_tau_t.device
                )
        if 'tpb_tau' in params:
            self.tpb_tau = params['tpb_tau']
        if 'light_oscillation_period' in params:
            self.light_oscillation_period = params['light_oscillation_period']
            with torch.no_grad():
                self.log_light_oscillation_period.data = torch.tensor(
                    np.log10(self.light_oscillation_period) / self.nominal_log_light_oscillation_period,
                    dtype=torch.float32,
                    device=self.log_light_oscillation_period.device
                )
        if 'light_response_time' in params:
            self.light_response_time = params['light_response_time']
            with torch.no_grad():
                self.log_light_response_time.data = torch.tensor(
                    np.log10(self.light_response_time) / self.nominal_log_light_response_time,
                    dtype=torch.float32,
                    device=self.log_light_response_time.device
                )
        if 'light_gain' in params:
            self.light_gain_value = params['light_gain']
            with torch.no_grad():
                self.light_gain.data = torch.tensor(
                    self.light_gain_value / self.nominal_light_gain,
                    dtype=torch.float32,
                    device=self.light_gain.device
                )
        if 'light_tick_size' in params:
            self.light_tick_size = params['light_tick_size']
        if 'downsample_factor' in params:
            self.downsample_factor = params['downsample_factor']
        if 'offset' in params:
            self.offset = params['offset']

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
        start_time = time.time()
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
        
        nonzero_indices = torch.nonzero(timing_dist)

        for (batch, det, tick) in nonzero_indices:
            n_photons = int(timing_dist[batch, det, tick].item())
            if n_photons == 0:
                continue

            # Sample singlet vs triplet for each photon
            singlet_fraction = self.singlet_fraction
            is_singlet = torch.rand((n_photons,), device=device) < singlet_fraction

            # Sample emission delays
            tau_s = self.tau_s
            tau_t = self.tau_t

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

        end_time = time.time()
        # print(f"total scintillation delay time: {end_time - start_time:.4f} sec")
        info['scintillation_delays'] = torch.cat(all_emission_delays)
    
        return delayed_counts, info

    def tpb_delay(self, timing_dist: torch.Tensor):
        '''
        Parameters:
            - timing_dist: Tensor of shape (nbatch, ndet, nticks) representing photon counts per time tick for each detector.
            - tau: lifetime of tpb re-emission spectrum
        Returns:
            - Tensor of shape (ndet, nticks) storing updated photon hit counts after applying scintillation delays.
        '''
        start_total = time.time()

        device = timing_dist.device
        if timing_dist.ndim < 3:
            timing_dist = timing_dist.unsqueeze(0)
        nbatch, ndet, nticks = timing_dist.shape
    
        # Define time ticks
        time_ticks = torch.arange(nticks, device=device)
        bin_width = self.light_tick_size  # Size of each time bin in microseconds

        all_emission_delays = []
        delayed_counts = torch.zeros_like(timing_dist) # in units of time ticks

        nonzero_indices = torch.nonzero(timing_dist)

        start_photon = time.time()
        for (batch, det, tick) in nonzero_indices:
            n_photons = int(timing_dist[batch, det, tick].item())
            if n_photons == 0:
                continue
            
            # sample
            emission_delays = torch.distributions.Exponential(1.0 / self.tpb_tau).sample((n_photons,)).to(device)
            all_emission_delays.append(emission_delays)

            base_time = tick * self.light_tick_size  # microseconds
            photon_times = base_time + emission_delays  # absolute photon times in microseconds

            bin_centers = time_ticks * self.light_tick_size
            bin_edges = torch.cat([
                bin_centers[:1] - bin_width / 2,  # left edge of first bin
                bin_centers + bin_width / 2      # right edges
            ])
            bin_indices = torch.bucketize(photon_times, bin_edges) - 1  # adjust for 0-based indexing

            # count photons in each bin
            valid_mask = (bin_indices >= 0) & (bin_indices < nticks)
            bin_indices = bin_indices[valid_mask]
            counts = torch.bincount(bin_indices, minlength=nticks)

            delayed_counts[batch, det] += counts
        
        end_photon = time.time()
        end_total = time.time()
        # print(f"Total tpb_delay time: {end_total - start_total:.4f} sec")
        # print(f"Photon delay and binning time: {end_photon - start_photon:.4f} sec")

        return all_emission_delays, delayed_counts

    def all_stochastic_sampling_vec(self, timing_dist:torch.Tensor) -> torch.Tensor:
        '''
        Combines stochastic scintillation + TPB re-emission delay sampling with full batching.
        Offers major speedups for high photon occupancy per (batch, det, tick).
        '''

        device = timing_dist.device
        if timing_dist.ndim < 3:
            timing_dist = timing_dist.unsqueeze(0)
        nbatch, ndet, nticks = timing_dist.shape
    
        bin_width = self.light_tick_size
    
        # finding nonzero photon time bins
        nz = torch.nonzero(timing_dist, as_tuple=False)  # shape (N, 3)
        n_photons_per_bin = timing_dist[nz[:, 0], nz[:, 1], nz[:, 2]].long()

        total_photons = n_photons_per_bin.sum().item()
        if total_photons == 0:
            delayed_counts = torch.zeros_like(timing_dist)
            return [], [], delayed_counts, {'num_singlets': 0, 'num_triplets': 0}

        # expanding photon coordinates for each photon
        photon_offsets = torch.repeat_interleave(nz, n_photons_per_bin, dim=0)
        batch_ids = photon_offsets[:, 0]
        det_ids = photon_offsets[:, 1]
        base_ticks = photon_offsets[:, 2]

        # vectorized sampling of scintillation and TPB delays
        with torch.no_grad():
            is_singlet = torch.rand((total_photons,), device=device) < self.singlet_fraction
            num_singlets = is_singlet.sum().item()
            num_triplets = total_photons - num_singlets
    
            scintillation_delays = torch.empty((total_photons,), device=device)
            scintillation_delays[is_singlet] = torch.empty((num_singlets,), device=device).exponential_(1.0 / self.tau_s)
            scintillation_delays[~is_singlet] = torch.empty((num_triplets,), device=device).exponential_(1.0 / self.tau_t)
    
            tpb_delays = torch.empty((total_photons,), device=device).exponential_(1.0 / self.tpb_tau)

            # computing total photon arrival time
            base_times = base_ticks * bin_width
            photon_times = base_times + scintillation_delays + tpb_delays
            photon_times = photon_times.double() 
            bin_indices = torch.floor(photon_times / bin_width).long()
            valid_mask = (bin_indices >= 0) & (bin_indices < nticks)
    
           # defining bin edges from bin centers
            time_ticks = torch.arange(nticks, device=device)
            bin_centers = time_ticks * bin_width
            bin_edges = torch.cat([
                bin_centers[:1] - bin_width / 2,
                bin_centers + bin_width / 2
            ])
            
            # bucketize to assign photons to bins
            photon_times = photon_times.double()  # increase precision to avoid edge artifacts
            bin_indices = torch.bucketize(photon_times, bin_edges) - 1  # adjust to 0-based indexing
            
            # removing photons outside the valid time bin range 
            valid_mask = (bin_indices >= 0) & (bin_indices < nticks)
            bin_indices = bin_indices[valid_mask]
            batch_ids = batch_ids[valid_mask]
            det_ids = det_ids[valid_mask]
                
            # making histogram w/ (batch, det, time) using index_add_
            delayed_counts = torch.zeros((nbatch, ndet, nticks), device=device)
            flat_idx = batch_ids * ndet * nticks + det_ids * nticks + bin_indices
            delayed_counts.view(-1).index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=delayed_counts.dtype))

        return (
            delayed_counts,
            {'num_singlets': num_singlets, 'num_triplets': num_triplets, 
             'scintillation_delays': scintillation_delays.tolist(), 'tpb_emission_delays': tpb_delays.tolist()}
        )

                    
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
            - nphotons (scalar): photons per arrival time
            - n_pmts: number of PMTs (default 128)
        Returns:
            - waveform: tensor of shape (n_pmts, n_ticks), where each row is a histogram of arrivals over time
            - from larnd sim, photon arrival times are offset by 2560 ticks. Default value here is 10 ticks (10 * 100 picoseconds = 1 nanosecond)
        '''
        assert pmt_ids.shape == arrival_times.shape
        arrival_times = arrival_times + self.offset # adding an offset

        waveform = torch.zeros((n_pmts, self.n_ticks))
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
        
        # for each pmt in pmt_ids
        # for each photon in nphotons[pmt_id]
        # sample arrival time from gaussian w std centered around 0
        waveform = torch.zeros((n_pmts, self.n_ticks))
        
        for i, pmt_id in enumerate(pmt_ids):
            pmt_arrival_times = torch.normal(mean=arrival_mean + self.offset, std=std * 1000, size=(nphotons[i],)) # offsetting arrival mean
            arrival_bins = pmt_arrival_times.round().long()
            
            # switching to no clamping to avoid edge artifacts
            valid_mask = (arrival_bins >= 0) & (arrival_bins < self.n_ticks)
            arrival_bins = arrival_bins[valid_mask]
            pmt_id_valid = torch.full_like(arrival_bins, pmt_id)
            
            waveform.index_put_((pmt_id_valid, arrival_bins),
                torch.ones_like(arrival_bins, dtype=waveform.dtype),
                accumulate=True)

        return waveform
    
    def sipm_response_model(self, time_tick, relax_cut=True) -> torch.Tensor:
        """
        Calculates the SiPM response from a PE at `time_tick` relative to the PE time

        Args:
            time_tick (torch.Tensor): time tick relative to t0
            relax_cut (bool): whether to apply the relaxing cut for differentiability

        Returns:
            torch.Tensor: response
        """
        start_time = time.time()
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

        end_time = time.time()
        return impulse * self.light_tick_size

    def fft_conv(self, light_sample_inc: torch.Tensor, model: Callable) -> torch.Tensor:
        """
        From LArND-Sim
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
        From LArND-Sim
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

    def downsample_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Downsample the input waveform by summing over groups of ticks.
        This effectively compresses the waveform in the time dimension while preserving the total signal.

        Args:
            waveform (torch.Tensor): Input waveform tensor of shape (n
            input, ndet, ntick), where each tick corresponds to 1 ns.
            self.downsample_factor is ns_per_tick (int, optional): Number of nanoseconds per tick in the downsampled waveform. Defaults to 16.

        Returns:
            torch.Tensor: Downsampled waveform of shape (ninput, ndet, ntick_down).
        """
        ninput, ndet, ntick = waveform.shape
        ntick_down = ntick // self.downsample_factor
        downsample = waveform.view(ninput, ndet, ntick_down, self.downsample_factor).sum(dim=3)
        return downsample
    
    def forward(self, timing_dist: torch.Tensor, scintillation=True, tpb_delay=True, combined=True, jax=False):
        '''
        Parameters:
            - timing_dist: Tensor of shape (nbatch, ndet, nticks)
        Returns:
            - Tensor of shape (nbatch, ndet, nticks) after processing.
        '''
        start_time = time.time()
        reshaped = False
        if timing_dist.ndim == 1:
            timing_dist = timing_dist[None, None, :]
            reshaped = True
        elif timing_dist.ndim == 2:
            timing_dist = timing_dist[None, :, :]
            reshaped = True
    
        nbatch, ndet, nticks = timing_dist.shape

        emission_delays = []

        x = timing_dist
        info = None

        if jax and combined:
            x, info = self.all_stochastic_sampling_vec(x)
            x = self.fft_conv_jax(x, partial(self.sipm_response_model_jax))
            x = self.light_gain * self.nominal_light_gain * x
            if reshaped:
                return x.squeeze(0).squeeze(0), info

            return x, info

        if combined:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_start = time.time()
            x, info = self.all_stochastic_sampling_vec(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_end = time.time()
            
        if scintillation and not combined:
            x, info = self.scintillation_model_sampled(x) # stochastic photon emission sampling
        if tpb_delay and not combined:
            tpb_emission_delays, x = self.tpb_delay(x) # stochastic re-emission from tpb
            info['tpb_emission_delays'] = torch.cat(tpb_emission_delays)

        x = self.fft_conv(x, partial(self.sipm_response_model)) # convolving with SiPM response kernel
        x = self.light_gain * self.nominal_light_gain * x
        # print(f"before downsampling: {x.shape}")
        x = self.downsample_waveform(x)
        # print(f"after downsampling: {x.shape}")
        
        end_time = time.time()
        # print(f"total forward time: {end_time - start_time:.4f} sec")
    
        if reshaped:
            return x.squeeze(0).squeeze(0), info
        
        return x, info
        
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
            (self.offset, self.n_ticks - output.shape[1] - self.offset),
            mode="constant",
            value=0
        )
        return output
    
    def batch_sample(self, nphoton: int, nbatch: int) -> torch.Tensor:
        return torch.stack([self(nphoton) for _ in range(nbatch)])


data_path = os.path.join(os.path.dirname(__file__), "data/lightLUT_Mod0_06052024_32.1.16_time_dist_cdf.npz")
data = np.load(data_path)
mod0_sampler = TimingDistributionSampler(**data)