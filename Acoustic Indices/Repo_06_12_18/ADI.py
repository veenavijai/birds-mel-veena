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