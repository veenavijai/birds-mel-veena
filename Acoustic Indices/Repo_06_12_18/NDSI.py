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