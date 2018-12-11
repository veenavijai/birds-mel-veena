#1. ACI
def compute_aci(specs, time_steps):
      
    import numpy as np
    #Function computes Acoustic Complexity Index
    #According to:
    #http://www.iinsteco.org/people/pubblications/almo/2011/2011_A_new%20methodology_to_infer_the_singing_activity.pdf
    
    #Adding eps
    eps = 1e-10
    specs = specs + eps  
    
    abs_diff = np.abs(specs[:, :, 1:] - specs[:, :, 0:time_steps-1])
    ACI = np.sum(np.sum(abs_diff, axis = 2)/np.sum(specs, axis = 2), axis = 1)
    
    return ACI
    

#2. Acoustic diversity
def compute_adi(specs):

    import numpy as np
    #Function computes Acoustic Diversity Index
    #According to: Pekin (2012)
    #Each bin corresponds to one of the 70 frequency points
   
    #Adding eps
    eps = 1e-10
    specs = specs + eps
    
    row_sums = specs.sum(axis=2, keepdims=False)
    all_sum = row_sums.sum(axis=1, keepdims=True)
    row_norm = row_sums/all_sum                                  #normalization
    ADI_all = (row_norm * np.log(row_norm)).sum(axis=1)

    return ADI_all
    
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
    

#3. Acoustic diversity with evenly spaced bins
def compute_adi_even(num_bins, specs, start_freq_arr, stop_freq_arr):

    import numpy as np
    #Function computes Acoustic Diversity Index
    #According to: Pekin (2012)
    
    #Adding eps
    eps = 1e-10
    specs = specs+eps
  
    num_specs = specs.shape[0]
    ADI_bins = np.zeros((num_specs, num_bins))
    
    ADI_all = 0
    
    for i in range(num_bins):      
    
        #for loop for all time steps in same freq bin - sum each row
        start = int(start_freq_arr[i])
        stop = int(stop_freq_arr[i])
        specs_strip = specs[:, start:stop, :]
        ADI_bins[:, i] = np.sum(np.sum(specs_strip, axis = 2), axis = 1)
        
    #Normalizing by summing along axis 1: bin number
    ADI_bins_norm = ADI_bins/np.sum(ADI_bins, axis = 1, keepdims=True)
    
    #Computing
    ADI_all = (ADI_bins_norm * np.log(ADI_bins_norm)).sum(axis=1)
    
    return ADI_all






#4. Spectral entropy 
def compute_sh(specs, no_freq_bins):
    
    import numpy as np
    #This function is a modified version of  spectral entropy as described in Sueur (2008)
    #Adding eps
    eps = 1e-10
    specs = specs+eps
    
    #Convert each frame to absolute value
    time_sums = np.sum(np.abs(specs), axis = 1, keepdims=True)
    spec_norm = np.divide(np.abs(specs), time_sums)

    #Added the eps in the log term, using np.multiply
    mult_term = np.multiply(spec_norm, np.log2(spec_norm + eps))
   
    sh_all = - np.sum(mult_term.sum(axis = 1), axis = 1)
    sh_all = sh_all/np.log2(no_freq_bins)          #normalizing
    
    return sh_all
    
    
#5. NDSI
def compute_ndsi(spec, start_freq_arr, stop_freq_arr, start_a_freq, stop_a_freq, start_b_freq, stop_b_freq):
    
    import numpy as np
    #Taken from the Kasten 2012 paper - page 6 has NDSI description
    
    #Adding eps
    eps = 1e-10
    spec = spec + eps
    
    #Anthrophony bin: 1-2 kHz
    start_a = int(start_freq_arr[start_a_freq])
    stop_a = int(stop_freq_arr[stop_a_freq-1])
    
    #Biophony bin: 2-8 kHz
    start_b = int(start_freq_arr[start_b_freq])
    stop_b = int(stop_freq_arr[stop_b_freq-1])
    
    #Taking absolute value - power spectral density was squared, so intensities are made positive here

    anth_sum = np.sum(np.sum(np.abs(spec[:, start_a:stop_a, :]), axis = 2), axis = 1)
    bio_sum = np.sum(np.sum(np.abs(spec[:, start_b:stop_b, :]), axis = 2), axis = 1)
    NDSI = (bio_sum-anth_sum)/(bio_sum+anth_sum)
    
    return NDSI
    
#function that individually calls the functions to calculate each acoustic index
def calc_ind(num_bins, spec_data, ch, start_freq_idx, stop_freq_idx, start_a_freq, stop_a_freq, start_b_freq, stop_b_freq):
    
    import numpy as np
    eps = 1e-10
    
    if(ch==0):
        #Raw spectrogram, so we take log
        all_specs = 10 * np.log10(spec_data['specs'][:, :, :] + eps)
    elif(ch==5):
        all_specs = spec_data
    else:
        ch=ch-1
        #4 differential channels. Last dimension is for ch_no. No log needed. 
        all_specs = spec_data['specs'][:, :, :, ch] 
    
    num_specs = all_specs.shape[0]
      
    ACI = np.zeros((num_specs, ))
    ADI = np.zeros((num_specs, ))
    ADI_even = np.zeros((num_specs, ))
    SH = np.zeros((num_specs, ))
    NDSI = np.zeros((num_specs, ))
               
    #Vectorized implementations 
    #ACI = compute_aci(all_specs, len(spec_data['spec_t']))
    ACI = compute_aci(all_specs, all_specs.shape[2])
    ADI = compute_adi(all_specs)
    ADI_even = compute_adi_even(num_bins, all_specs, start_freq_idx, stop_freq_idx)
    SH = compute_sh(all_specs, all_specs.shape[1])
    #SH = compute_sh(all_specs, len(spec_data['spec_f']))
    NDSI = compute_ndsi(all_specs, start_freq_idx, stop_freq_idx, start_a_freq, stop_a_freq, start_b_freq, stop_b_freq)
           
    return ACI, ADI, ADI_even, SH, NDSI