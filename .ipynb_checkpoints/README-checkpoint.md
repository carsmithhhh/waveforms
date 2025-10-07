# waveforms
Waveform simulator for LArTPC PMTs

Based off of waveform simulator from `larnd-sim`, but includes stochastic processes for scintillation delay and TPB wavelength shifter re-emission. Also includes functionality to generate pre-simulation waveforms as inputs.

**Waveform simulator steps:**
* Input is N photons and timing
* Each photon gets a delay sampled from a prompt or delayed scintillation light distribution (dictated by `singlet_fraction` parameter)
* Each photon gets a delay sampled from tpb re-emission timing distribution with lifetime of 2 nanoseconds
* Electronics response convolution from `larnd-sim` applied at the end

**Notebooks:**
* Look at `00_input_gen.ipynb` to use waveform generator --> only photon & timing information per waveform
* Look at `01_stochastic_simulator.ipynb` to apply simulation --> delay sampling and electronics conv.
