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