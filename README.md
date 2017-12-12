# Environmental-sound-recognition-using-combination-of-spectrogram-and-acoustic-features
Classification of environmental sounds using first order statistics and GLCM (Gray-Level Co-Occurrence Matrix ) features of a spectrogram 

Please refer to the paper below for more information of the algorithm and results
http://esatjournals.net/ijret/2017v06/i10/IJRET20170610015.pdf

## Requirements

### Python 2.7.10

### Python Modules
1.  Librosa 0.4.3
2.  numpy 1.11.3
3.  sklearn 0.18
4.  matplotlib 1.4.3
5.  SimpleITK 1.0.0
6.  pyradiomics 1.3.0



## Steps 

### Preprocessing
1.  Resampling the audio audio to 24,000 Hz and applying a high pass filter with a cut off frequency of 500Hz to remove the low frequency noise in the audio signals. 
2.  Compute the decibel scaled spectrogram image of the audio. Calculate the spectrogram with an FFT size of 512 which gives a frequency resolution of 46.875 Hz corresponding to the sampling rate of 24,000Hz. 
3.  For the hanning window,  a window length of 20ms with 75% overlap is used. 
4.  Rescale the spectrogram to a maximum value of 255. Figure-1(a) shows a spectrogram of a dog bark rescaled to amplitude in the range [0,255]
5.  On the rescaled image, use k-means with ten cluster centers to vector quantize the image to ten levels. 
6.  Perform binary thresholding with the threshold being the second highest value among the cluster centers. Figure-1(b) shows spectrogram image quantized to ten levels.  
7.  Create a binary mask and retain the part of the image which correspond to two cluster centers with the highest pixel values. Figure-1(c) shows the connected components extracted from the binary mask obtained after thresholding. 
8.  By considering the left most and right most location of each of the connected component, extract corresponding segments in the time domain. Figure1(d) shows the obtained segments in the time domain 
9.  Extract the prominent part of the signal.  Figure1(e) shows  the extracted prominent part of the signal.

### Feature Extraction

#### **First Set of Features**
1.  Divide the obtained spectrogram of the image and divide into four equal frequency bands (sub bands).
2.  Compute first order statistics and glcm features for each of the sub bands.

##### **First order statistics**
Minimum,mean, median, variance, energy, entropy, tenth percentile pixel value, ninetieth percentile pixel value, inter quartile range, mean absolute deviation, robust mean absolute deviation, root mean square error, skewness and kurtosis. 

##### **GLCM features (Combination of angles (0,45,90,135), displacement vectors (3,5))**
Energy, contrast, correlation, sum of squares, inverse of difference moment, sum average, sum entropy, sum variance, entropy, difference variance, difference entropy, and two descriptors of information measure of correlation.

#### **Second Set of Features**
Extracted with window length of 20ms and 75% overlap between frames 

MFCCs (Mel Frequency Cepstral Coefficients), Delta MFCCs, ZCR (Zero Crossing Rate), RMSE (Root Mean Square Error), spectral centroid, spectral bandwidth, spectral contrast and spectral rolloff 

### Classification

Four different models

-  SIF (Spectrogram Image Features) Model
    - Trained Separately on **First Set of Features**
-  AF  (Acoustic Features) Model
    - Trained Separately on **Second Set of Features**
-  ASIF  (Acoustic and Spectrogram Image Features) Model
    - Trained with both  **First Set of Features** and **Second Set of Features** combined in the feature space.
-  MEASIF (Modified Ensemble of Acoustic and Spectrogram Image Features) Model
    - Modified Ensemble of SIF and AF models. Figure-3 shows the architecture of MEASIF Model

## Results
To Evaluate the approach, ESC-10 dataset available at https://github.com/karoldvl/ESC-10 was used.

ESC-50 is a dataset with annotated collection of 2,000 short clips comprising 50 classes of various common sound events. Each class consists of 40 sound clips with each sound clip 5-seconds-long reconverted into a unified format (44.1 kHz, single channel, Ogg Vorbis compression at 192 kbit/s). The labeled datasets were consequently arranged into 5 uniformly sized cross-validation folds.

The ESC-10 is a selection of 10 classes from the bigger dataset ESC-50.

The Frieburg-106 dataset was collected using a consumer level dynamic cardioid microphone. It contains 1,479 audio based human activities of 22 categories.


