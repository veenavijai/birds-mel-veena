## Classification of spectrograms into bird calls vs background noise vs human speech

This repo documents my work with Dr. Shyam Madhusudhana as part of my internship with the Bioacoustics Research Program, Cornell Lab of Ornithology.

**indices_file.py

My implementations of 5 acoustic indices from [this](http://www.soundandlightecologyteam.colostate.edu/pdf/ecoacoustics2018.pdf) paper. They are: 
1. Acoustic Complexity Index (ACI)
2. Acoustic Diversity Index (ADI)
3. Acoustic Diversity Index for even bins (ADI_even)
4. Spectral Entropy (SH)
5. Normalized Difference Soundscape Index (NDSI)

**load_files.py

Loads all the spectrogram data which is stored as a dictionary.

**scatter_all_exp4_5.py

This program loads data from 4 different classes - 'good' (probably contains a bird call), bad (probably just background noise), human (human speech) \& maybe (doubtful what the content is). For each spectrogram, it calculates the 5 above acoustic indices, and uses vectorized implementations wherever possible. The final result is a scatterplot for each acoustic index along with a scatterplot matrix made in seaborn.

**scatter_10files_exp4_5.py

This program plots scatterplot matrices for 10 clips downloaded from the internet. Each clip is 2 seconds long and considered to be a distinct class. 

File name list: ["Airplane_Sound", "Heavy_rain", "Brown_Noise", "Pink_Noise", "White_Noise", 
#"Rufous_Antpitta", "Grey_headed_woodpecker", "Italian_Sparrow", "Hawk_scream", "Dove"]

