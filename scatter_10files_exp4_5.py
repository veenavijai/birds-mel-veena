#Instead of 4 classes - we have 10 classes.

#Plotting scatterplot matrix for the 10 files clips downloaded from the internet
#Each clip is 2s long

#Uses most recently updated scatterplot matrix code
#From Scatterplot_Experiments_till_5_Modified and Scatterplot_Experiments_till_5_pt_1

#Also uses code for 10 files from ACI_check

#fname_list = ["Airplane_Sound", "Heavy_rain", "Brown_Noise", "Pink_Noise", "White_Noise", 
#"Rufous_Antpitta", "Grey_headed_woodpecker", "Italian_Sparrow", "Hawk_scream", "Dove"]

#For each file, we will return - 
#(1) from calls_orig - an array for each Acoustic Index with as many elements as the number of spectrograms extracted from it
#(num_specs will be set accordingly)
#(2) from calls - 4 arrays - one for each channel - and 5 for each index (so 20). No. of elements according to num_specs


#General imports
import numpy as np
from matplotlib import pyplot as plt
import os

#For scatter matrixs
import pandas as pd
from pandas.plotting import scatter_matrix

#Seaborn for visualizations
import seaborn as sns
sns.set(style="ticks")

#TODO
#imports for Acoustic indices functions
from indices_file import compute_aci, compute_adi, get_start_stop_indices, compute_adi_even, compute_sh, compute_ndsi, calc_ind
from load_files import load_display_dimensions

#General parameters

eps = 1e-10                #setting eps value for converting spec to log
dim = 0                    #whether or not to display dimensions
str_file = '_melspec.npz'  #if we're using mel spectrogram
#str_file = '_spec.npz'    #if we're using raw spectrogram

#This means, we want to find the indices on the frequency axis, for a spacing of 1 kHz and 8 bins. 
#Like, 0-1 kHz, 1-2 kHz, and so on, up to 7-8 kHz.
num_bins = 8            #for compute_adi_even, NDSI
multiples = 1000        #spacing to calculate frequency indices

#NDSI bin frequencies in kHz according to the original paper
#a = anthrophony bin, 1-2 kHz
#b = biophony bin, 2-8 kHz
start_a_freq = 1
stop_a_freq = 2
start_b_freq = 2
stop_b_freq = 8

#file list
fname_list = ["Airplane_Sound", "Heavy_rain", "Brown_Noise", "Pink_Noise", "White_Noise", "Rufous_Antpitta", "Grey_headed_woodpecker", "Italian_Sparrow", "Hawk_scream", "Dove"]
num_files = len(fname_list)

#calls all functions for each class - is for the original data
def calls_channels(class_name, multiples, num_bins, dim, start_a_freq, stop_a_freq, start_b_freq, stop_b_freq):

    #Load the correct files
    spec_data_ch0 = load_display_dimensions(class_name, str_file, dim)
    
    str_file2 = '_melspec_dd.npz'
    spec_data_all4 = load_display_dimensions(class_name, str_file2, dim)       
       
    #Calling helper function for compute_adi_even
    start_freq_arr_ch0, stop_freq_arr_ch0 = get_start_stop_indices(spec_data_ch0['spec_f'], multiples, num_bins)
    start_freq_arr_all4, stop_freq_arr_all4 = get_start_stop_indices(spec_data_all4['spec_f'], multiples, num_bins)

    '''Check
    print("Start", start_freq_arr)
    print("Stop", stop_freq_arr)
    print(spec_data['spec_f'])
    '''
    
    #Initializing lists
    ACI = []
    ADI = []
    ADI_even = []
    SH = []
    NDSI = []
    
    #Computing acoustic indices for ch0
    ch=0
    ACI1, ADI1, ADI_even1, SH1, NDSI1 = calc_ind(num_bins, spec_data_ch0, ch, start_freq_arr_ch0, stop_freq_arr_ch0, start_a_freq, stop_a_freq, start_b_freq, stop_b_freq)

    #Appending to the list
    ACI.append(ACI1)
    ADI.append(ADI1)
    ADI_even.append(ADI_even1)
    SH.append(SH1)
    NDSI.append(NDSI1)
    
    #Computing acoustic indices for all 4 channels
    for i in range(4):
        
        #Computing acoustic indices for ith channel
        ch = i+1       #ch = 1,2,3,4
        ACI1, ADI1, ADI_even1, SH1, NDSI1 = calc_ind(num_bins, spec_data_all4, ch, start_freq_arr_all4, stop_freq_arr_all4, start_a_freq, stop_a_freq, start_b_freq, stop_b_freq)
        
        #Appending to the list
        ACI.append(ACI1)
        ADI.append(ADI1)
        ADI_even.append(ADI_even1)
        SH.append(SH1)
        NDSI.append(NDSI1)
    
    #Now the ACI list has 5 arrays. The ith array in the list has the ACI values for all spectrograms, for the ith channel.
    
    return ACI, ADI, ADI_even, SH, NDSI
    

#Initializing list of lists
ACI = []
ADI = []
ADI_even = []
SH = []
NDSI = []

#Calls file by file
for file_no in range(num_files):
    
    ACI_file, ADI_file, ADI_even_file, SH_file, NDSI_file = calls_channels(fname_list[file_no], multiples, num_bins, dim, start_a_freq, stop_a_freq, start_b_freq, stop_b_freq)
    #Now we have the 5 acoustic index values for all specs in file (file_no), for all 5 channels, stored in xx_file
    
    #Appending to the list
    ACI.append(ACI_file)
    ADI.append(ADI_file)
    ADI_even.append(ADI_even_file)
    SH.append(SH_file)
    NDSI.append(NDSI_file)
    
    
#General procedure to get pandas dataframes: 
#(1)reshape the array
#(2) create a dataframe with columns as acoustic indices
#(3) add an extra column with class name

def convert_df(ACI, ADI, ADI_even, SH, NDSI, class_name, ch):
   
    data = pd.DataFrame({'ACI': ACI, 'ADI': ADI, 'ADI_even': ADI_even, 'SH': SH,'NDSI': NDSI, 'Class': class_name}, columns=['ACI', 'ADI', 'ADI_even', 'SH', 'NDSI', 'Class'])
    
    #renaming the columns of the dataframe 'data' to be used while combining not NaN rows later
    if(ch!=0 and ch!=14 and ch!=15):
        data.rename(columns={'ACI': 'ACI'+str(ch), 'ADI': 'ADI'+str(ch), 'ADI_even': 'ADI_even'+str(ch), 'SH': 'SH'+str(ch), 'NDSI': 'NDSI'+str(ch), 'Class': 'Class'+str(ch)}, inplace=True)
 
    return data
    
    
def get_df_and_plot(ch, ACI, ADI, ADI_even, SH, NDSI):
    
    all_data_arr = []
    
    for i in range(num_files):
        
        #Converting data to dataframes: doing for a specific channel ch, for the ith file
        data_file = convert_df(ACI[i], ADI[i], ADI_even[i], SH[i], NDSI[i], fname_list[i], ch)
        all_data_arr.append(data_file)

    #Converting to a pandas dataframe
    all_data_df = pd.concat(all_data_arr)
    
    #Scatterplot matrix for all 4 classes

    if(ch!=0 and ch!=14 and ch!=15):
        Hue = 'Class' + str(ch)
    else:
        Hue = 'Class'
    
    #Setting seaborn specifications

    #Using a handmade palette - first 5 classes are red-based (noise), next 5 are blue-based (birds)
    custom_palette = ["hotpink", "red", "saddlebrown", "crimson", "darkorange", "blue", "dodgerblue", "navy", "springgreen", "aqua"]
    sns.set_palette(custom_palette)

    #plot_kws adjusts marker size
    a = sns.pairplot(all_data_df, hue=Hue, plot_kws={"s": 20}, diag_kind="kde")
    
    #Title
    
    if(ch==0):
        plot_title = "Scatterplot matrix for spectrogram"
    elif(ch==14):
        plot_title = "Scatterplot matrix for maximum of channels"
    elif(ch==15):
        plot_title = "Scatterplot matrix for average of channels"
    else:
        plot_title = "Scatterplot matrix for channel" + str(ch)
        
    a.fig.suptitle(plot_title)
    a.fig.subplots_adjust(top=.9)
    
    plt.show()
    
    return all_data_df
    
#All plots
all_data_df = []

for j in range(5):
    
    #Extracts from the 10 files (0-9, set by the i counter), the channel we want (0-4, set by the j counter)
    ACI_channel = [i[j] for i in ACI]
    ADI_channel = [i[j] for i in ADI]
    ADI_even_channel = [i[j] for i in ADI_even]
    SH_channel = [i[j] for i in SH]
    NDSI_channel = [i[j] for i in NDSI]
    
    #j = channel number (0 for spec, 1-4 for corresponding channels)
    df_returned = get_df_and_plot(j, ACI_channel, ADI_channel, ADI_even_channel, SH_channel, NDSI_channel)
                                  
    all_data_df.append(df_returned)
    
    
def combine_specs_4_5(exp, spec_data):
    
    if(exp==4):
        spec_1 = spec_data['specs'].max(axis=3, keepdims=False)
        
    elif(exp==5):
        spec_1 = spec_data['specs'].mean(axis=3, keepdims=False)

    return spec_1
    
    
#calls all functions for each class - is for Exp4
def calls_exp_4_5(exp, fname, multiples, num_bins, dim, start_a_freq, stop_a_freq, start_b_freq, stop_b_freq):

     #Load the correct files
    str_file2 = '_melspec_dd.npz'
    spec_data_all4 = load_display_dimensions(fname, str_file2, dim) 
       
    #Calling helper function for compute_adi_even
    start_freq_arr_all4, stop_freq_arr_all4 = get_start_stop_indices(spec_data_all4['spec_f'], multiples, num_bins)
    
    #Combining spectrograms
    new_spec = combine_specs_4_5(exp, spec_data_all4)
    
    #Computing acoustic indices for all 4 channels 
    ch = 5
       
    ACI, ADI, ADI_even, SH, NDSI = calc_ind(num_bins, new_spec, ch, start_freq_arr_all4, stop_freq_arr_all4, start_a_freq, stop_a_freq, start_b_freq, stop_b_freq)
        
    return ACI, ADI, ADI_even, SH, NDSI
    
#Calls for each class - Exp 4 with max values

exp = 4
#Initializing list of lists
ACI_4 = []
ADI_4 = []
ADI_even_4 = []
SH_4 = []
NDSI_4 = []

#Calls file by file
for file_no in range(num_files):
    
    ACI_file, ADI_file, ADI_even_file, SH_file, NDSI_file = calls_exp_4_5(exp, fname_list[file_no], multiples, num_bins, dim, start_a_freq, stop_a_freq, start_b_freq, stop_b_freq)
    #Now we have the 5 acoustic index values for all specs in file (file_no), for all 5 channels, stored in xx_file

    #Appending to the list
    ACI_4.append(ACI_file)
    ADI_4.append(ADI_file)
    ADI_even_4.append(ADI_even_file)
    SH_4.append(SH_file)
    NDSI_4.append(NDSI_file)
    
#Plots
ch = 14 #needed for distinct plot title
df_returned_4 = get_df_and_plot(ch, ACI_4, ADI_4, ADI_even_4, SH_4, NDSI_4)


#Calls for each class - Exp 5 with avg values

exp = 5
#Initializing list of lists
ACI_5 = []
ADI_5 = []
ADI_even_5 = []
SH_5 = []
NDSI_5 = []

#Calls file by file
for file_no in range(num_files):
    
    ACI_file, ADI_file, ADI_even_file, SH_file, NDSI_file = calls_exp_4_5(exp, fname_list[file_no], multiples, num_bins, dim, start_a_freq, stop_a_freq, start_b_freq, stop_b_freq)
    #Now we have the 5 acoustic index values for all specs in file (file_no), for all 5 channels, stored in xx_file

    #Appending to the list
    ACI_5.append(ACI_file)
    ADI_5.append(ADI_file)
    ADI_even_5.append(ADI_even_file)
    SH_5.append(SH_file)
    NDSI_5.append(NDSI_file)
    
#Plots
ch = 15 #needed for distinct plot title
df_returned_5 = get_df_and_plot(ch, ACI_5, ADI_5, ADI_even_5, SH_5, NDSI_5)
    