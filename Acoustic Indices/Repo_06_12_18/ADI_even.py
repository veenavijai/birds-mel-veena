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