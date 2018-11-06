#This program loads data from 4 different classes
#good, bad, human, maybe
#And uses 10 different acoustic indices on num_spec recordings from each
#Each recording is exactly 2 s long
#Final result: a scatter plot for each acoustic index
#Values on x axis: values of acoustic index
#Colours: for each class
#List of acoustic indices used: 
#1. ACI 2. Roughness 3. Temporal entropy 4. Acoustic diversity 5. Acoustic richness 
#6. Spectral diversity 7. Spectral kurtosis 8. Spectral entropy 9. ?? 10. ??

#imports
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import hilbert
from scipy.signal import stft
import librosa
import os

#for scatter matrix
import pandas as pd
from pandas.plotting import scatter_matrix

#setting eps value for converting spec to log
eps = 1e-10



#function to load spectrograms and display dimensions (optional)
def load_display_dimensions(class_name, dim=0):

    #   description of data files
    #   spec_data is like a dict with fields
    #   'specs' (contains the mel-filtered spectrograms, in linear scale, as [num_specs, num_rows, num_cols])
    #   'spec_f' (contains the frequency axis points as a 1D array)
    #   'spec_t' (contains the time axis points as a 1D array)   
    
    fname_load = class_name + '_preprocessed_uncompressed.npz'
    spec_data = np.load(fname_load)
    
    time_steps = len(spec_data['spec_t'])
    freq_bins = len(spec_data['spec_f'])
    
    if (dim==1):
        print('File has %i specs of dimensions (%i x %i)' % (
        spec_data['specs'].shape[0], spec_data['specs'].shape[1], spec_data['specs'].shape[2]))
        print('Frequency axis points are in spec_data[\'spec_f\'] and has %i values' % freq_bins)
        print('Time axis points are in spec_data[\'spec_t\'] and has %i values' % time_steps)
        
    return spec_data, time_steps, freq_bins
    
dim = 1
spec_data, time_steps, freq_bins = load_display_dimensions('good', dim)
#spec_data, time_steps, freq_bins = load_display_dimensions('bad', dim)
#spec_data, time_steps, freq_bins = load_display_dimensions('human', dim)
#spec_data, time_steps, freq_bins = load_display_dimensions('maybe', dim)

spec_idx = 50
spec0 = spec_data['specs'][spec_idx, :, :, 0] 
spec1 = spec_data['specs'][spec_idx, :, :, 1] 
spec2 = spec_data['specs'][spec_idx, :, :, 2] 
spec3 = spec_data['specs'][spec_idx, :, :, 3] 

#loading original files as well
class_name = 'good'
fname_load = class_name + '.npz'
spec_data2 = np.load(fname_load)

eps = 1e-10
orig_spec = 10 * np.log10(spec_data2['specs'][spec_idx, :, :] + eps) 

fig, axes = plt.subplots(5, 1, sharex='all')
plt.title('Spectrograms for ' + 'good')

time_steps = len(spec_data['spec_t'])
freq_bins = len(spec_data['spec_f'])

preproc_cmap = plt.get_cmap('winter_r')
preproc_cmap.set_under('w')         #setting values below the lower limit to appear white


#plotting original
cax = axes[0].imshow(orig_spec, cmap=plt.get_cmap('jet'), interpolation='none', origin='lower')
#cax.set_clim(vmin = np.finfo(np.float32).tiny)
fig.colorbar(cax, ax=axes[0], orientation='vertical')

#plotting differentials

cax = axes[1].imshow(spec0, cmap=plt.get_cmap('winter_r'), interpolation='none', origin='lower')
cax.set_clim(vmin = np.finfo(np.float32).tiny)
fig.colorbar(cax, ax=axes[1], orientation='vertical')

cax = axes[2].imshow(spec1, cmap=plt.get_cmap('winter_r'), interpolation='none', origin='lower')
cax.set_clim(vmin = np.finfo(np.float32).tiny)
fig.colorbar(cax, ax=axes[2], orientation='vertical')

cax = axes[3].imshow(spec2, cmap=plt.get_cmap('winter_r'), interpolation='none', origin='lower')
cax.set_clim(vmin = np.finfo(np.float32).tiny)
fig.colorbar(cax, ax=axes[3], orientation='vertical')

cax = axes[4].imshow(spec3, cmap=plt.get_cmap('winter_r'), interpolation='none', origin='lower')
cax.set_clim(vmin = np.finfo(np.float32).tiny)
fig.colorbar(cax, ax=axes[4], orientation='vertical')

plt.show()