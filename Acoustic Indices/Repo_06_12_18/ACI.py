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