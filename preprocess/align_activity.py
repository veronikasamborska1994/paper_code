import numpy as np
import pylab as plt
from scipy.stats import norm

def align_activity(trial_times, target_times, spikes, fs=25, smooth_SD='default', plot=False):
    '''Calculate trial aligned smoothed firing rates. Spike times are first transformed from 
    the original time frame to a trial aligned time frame in which a set of reference time
    points for each trial are mapped onto a set of target time points (e.g. the median trial
    timings), with linear interpolation of spike times between the reference points.  
    Once the spike times have been transformed into the trial aligned reference frame the
    firing rate is calculated at a specified sampling rate, using Gaussian smoothing with 
    a specified standard deviation.

    Arguments:
    trial_times : Array of reference point times for each trial (ms). Shape: [n_trials, n_ref_points]
    target_times: Reference point times to warp each trial onto (ms). Shape: [n_ref_points]
    spikes:  Array of neuron IDs and spike times. Shape [, n_spikes]
             spikes[0,:] is neuron IDs, spikes [1,:] is spike times (ms).
    fs: Sampling rate of output firing rate vector (Hz).
    smooth_SD: Standard deviation of gaussian smoothing applied to ouput rate (ms). 
               If set to default, smooth_SD is set to the inter sample interval.
    plot: If set to True, plots the average trial aligned activity for first 5 neurons.

    Returns:
    aligned_rates: Array of trial aligned smoothed firing rates (Hz). 
                   Shape: [n_trials, n_neurons, n_timepoints]
    t_out: Times of each output firing rate time point (ms).
    min_max_stretch: Minimum and maximum stretch factor for each trial.  Used to exclude 
                     trials which have extreme deviation from target timings.
                     Shape: [n_trials, 2]
    '''
    if smooth_SD == 'default': smooth_SD = 1000/fs
    n_trials = trial_times.shape[0]
    neuron_IDs = sorted(list(set(spikes[0,:])))
    n_neurons = len(neuron_IDs)
    t_out = np.arange(target_times[0], target_times[-1]+1000/fs, 1000/fs) # Times of output samples in target reference frame (ms).
    # Add interval before and after specified intervals to prevent edge effects.
    target_times = np.hstack([target_times[0]-3*smooth_SD, target_times, target_times[-1]+3*smooth_SD])
    trial_times = np.hstack([trial_times[:,0,None]-3*smooth_SD, trial_times, trial_times[:,-1,None]+3*smooth_SD])
    # Apply timewarping.
    target_deltas = np.diff(target_times) # Intervals between target time points.
    trial_deltas = np.diff(trial_times,1) # Intervals between reference points for each trial.
    stretch_factors = target_deltas/trial_deltas # Amount each interval of each trial must be stretched/squashed by.
    min_max_stretch = np.vstack([np.min(stretch_factors,1), np.max(stretch_factors,1)]).T # Minimum and maximum stretch factor for each trial.
    aligned_rates = np.zeros([n_trials, n_neurons, len(t_out)]) # Array to store trial aligned firing rates. 
    for t in range(n_trials): # Loop over trials.
        trial_spikes = spikes[:, (trial_times[t,0] < spikes[1,:]) & 
                                 (spikes[1,:] < trial_times[t,-1])]
        # Change times of trial_spikes to map them onto target_times.
        spike_stretch = np.zeros(trial_spikes.shape[1]) # Stretch factor for each spike.
        for i in range(len(target_times)-1): # Loop over intervals.
            interval_mask = ((trial_times[t,i] < trial_spikes[1,:]) &  # Boolean mask indicating which spikes  
                             (trial_spikes[1,:] < trial_times[t,i+1])) # in trial_spikes are in interval i.
            trial_spikes[1,interval_mask] = target_times[i] + stretch_factors[t,i] * (
                                            trial_spikes[1,interval_mask] - trial_times[t,i])
            spike_stretch[interval_mask] = stretch_factors[t,i]
        for j, n in enumerate(neuron_IDs): # Loop over neurons.
            if n in trial_spikes[0,:]:
                neuron_mask = trial_spikes[0,:] == n
                n_spike_times = trial_spikes[1,neuron_mask]
                aligned_rates[t,j,:] = 1000*np.sum(norm.pdf(n_spike_times[None,:]
                    -t_out[:,None], scale=smooth_SD)*spike_stretch[neuron_mask],1)
    if plot: # Plot trial aligned activity for the first 5 neurons.
        plt.figure(1).clf()
        for n in range(5):#most_active:
            plt.plot(t_out,np.mean(aligned_rates[:,n,:],0))
        for t in target_times[2:-2]:
            plt.axvline(t, color='k', linestyle=':')
        plt.xlim(t_out[0], t_out[-1])
        plt.ylim(ymin=0)
        plt.xlabel('Aligned time (ms)')
        plt.ylabel('Firing rate (Hz)')
    return aligned_rates, t_out, min_max_stretch