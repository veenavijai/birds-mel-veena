#function that individually calls the functions to calculate each acoustic index
def calc_ind(num_bins, spec_data, ch, start_freq_idx, stop_freq_idx, start_a_freq, stop_a_freq, start_b_freq, stop_b_freq):
    
    import numpy as np
    eps = 1e-10
    
    num_specs = spec_data['specs'].shape[0]
    
    ACI = np.zeros((num_specs, ))
    ADI = np.zeros((num_specs, ))
    ADI_even = np.zeros((num_specs, ))
    SH = np.zeros((num_specs, ))
    NDSI = np.zeros((num_specs, ))
               
    if(ch==0):
        #Raw spectrogram, so we take log
        all_specs = 10 * np.log10(spec_data['specs'][:, :, :] + eps)
    else:
        ch=ch-1
        #4 differential channels. Last dimension is for ch_no. No log needed. 
        all_specs = spec_data['specs'][:, :, :, ch] 

    #Vectorized implementations 
    ACI = compute_aci(all_specs, len(spec_data['spec_t']))
    ADI = compute_adi(all_specs)
    ADI_even = compute_adi_even(num_bins, all_specs, start_freq_idx, stop_freq_idx)
    SH = compute_sh(all_specs, len(spec_data['spec_f']))
    NDSI = compute_ndsi(all_specs, start_freq_idx, stop_freq_idx, start_a_freq, stop_a_freq, start_b_freq, stop_b_freq)
           
    return ACI, ADI, ADI_even, SH, NDSI