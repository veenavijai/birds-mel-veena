#Helper function for compute_adi_even
def get_start_stop_indices(freq_bins, multiples=1000, num_bins=8):
    
    import numpy as np
    #find the frequency index numbers to group in bins of 1 kHz width from 0-1 kHz, 1-2, till 7-8kHz
    stop_freq_arr = np.zeros((num_bins, ))
    start_freq_arr = np.zeros((num_bins, ))
    
    #So, if we want 3-4 kHz, we take start_idx = start_freq_arr[3] and stop_idx =stop_freq_arr[3]
    #freq_vals = spec_data['spec_f'][start_idx] and spec_data['spec_f'][stop_idx]
    #When we sum over those values, we use - spec_data[:, start_idx:stop_idx, :] as done in compute_adi_even

    for i in range(num_bins):
        stop_freq = multiples*(i+1)
        #to find the index to clip
        stop_freq_arr[i] = freq_bins.searchsorted(stop_freq, side='right') - 1
        if (i<num_bins-1):
            start_freq_arr[i+1] = stop_freq_arr[i]+1
        
    return start_freq_arr, stop_freq_arr