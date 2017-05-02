import numpy as np
import librosa
from scipy import interpolate
import pywt
from matplotlib.image import imsave
from scipy.signal import butter, lfilter, freqz
from matplotlib import pyplot as plt
from imageProcessingUtil import ImageProcessing
import SimpleITK as sitk



class AudioProcessing(object):

    def __init__(self):
        pass

    @staticmethod
    def read(absFilePath,sr=None):
        """
        Reading audio
        
        :param absFilePath: Absolute File Path
        :param sr: Sampling rate of audio to be read (If None, original sampling rate is considered)
        :return: audio samples, 
        """
        data,fs = librosa.load(absFilePath,sr=sr)
        return data,fs

    @staticmethod
    def writeAsWav(data,sr,filename):
        """
        Write .wav files 
        :param data: audio data
        :param sr: sampling rate
        :param filename: filename to be saved
        :return: None
        """

        if filename is None or sr is None or data is None :
            return "Please provid arguements as writeAsWav(data,sr,filename)"

        if "wav" not in filename:
            return "Only wav files!"

        filename_split = filename.rsplit(".",1)
        filename = filename_split[0]
        filetype = filename_split[1].lower()

        data = AudioProcessing.rescaleAmplitude(data)

        librosa.output.write_wav("{}.{}".format(filename,filetype),data,sr)

    @staticmethod
    def generateSineWave(amp,f,phi,fs):
        """
        Generating a simple sine wave 
        :param amp: Amplitude
        :param f: Frequency
        :param phi: Phase
        :param fs: Frequency sampling rate
        :return: Sine wave signal
        """
        # considering 5 time periodics
        t = np.arange(0,10.0/f,1.0/fs)
        x = amp*np.cos(2*np.pi*f*t + phi)

        return(t,x)


    @staticmethod
    def convert_to_mono(x):
        """
        Convert multi channel sounds to mono channel
        :param x: audio data
        :return: mono channel (audio data)
        """
        if x.ndim > 1:
            return librosa.to_mono(x)
        return x

    @staticmethod
    def DFT(data,N,fs,start_time = 0.0):


        """
        calculating N point DFT
        :param data: audio data
        :param N: N point DFT
        :param fs: sampling frequency
        :return:
        """
        data = AudioProcessing.convert_to_mono(data)

        size = data.size

        new_data = np.zeros(N)
        if size < N:
            diff = N - size
            new_data[:size] = data
        else:
            new_data = data[start_time*fs:start_time*fs+N]

        hanning = np.hanning(N)

        new_data = new_data*hanning

        print("Calculating DFT for {} ms window with start time {} sec".format(N*1000/float(fs),start_time))

        nv = np.arange(N)
        kv = np.arange(N)


        nv = np.arange(-N/2.0,N/2.0)
        kv = np.arange(-N/2.0,N/2.0)

        X = np.array([])

        # Calculating the DFT of the cropped signal
        for k in kv:
            s = np.exp(1j*2*np.pi*k/N*nv)
            X = np.append(X,sum(new_data*np.conjugate(s)))

        X = np.abs(X)
        frequency_axis = kv*fs/N

        return (frequency_axis,X)


    @staticmethod
    def resampleAudio(data,fs,new_fs):
        """
        Resampling audio to a different sampling rate
        :param data: audio data
        :param fs: old sampling rate
        :param new_fs: new sampling rate
        :return: resampled audio 
        """
        print("Resampling from {} to {} hz".format(fs,new_fs))

        fs = float(fs)
        new_fs = float(new_fs)

        data = AudioProcessing.convert_to_mono(data)
        size = data.size

        old_time_axis = np.arange(size)/fs

        total_time = old_time_axis[-1]

        total_samples = round(total_time*new_fs)

        new_time_axis = np.arange(total_samples)/new_fs

        f = interpolate.interp1d(old_time_axis,data)
        new_data = f(new_time_axis)
        return new_data

    @staticmethod
    def rescaleAmplitude(data,scale_range = (-1,1)):
        """
        rescaling an array to a particlar range 
        :param data:  Any array 
        :param scale_range: The range to which rescaling has to be done 
        :return: rescaled array
        """
        mini = np.min(data)
        maxi = np.max(data)

        new_min = scale_range[0]
        new_max = scale_range[1]

        new_data = ((new_max - new_min)*(data - mini)/(maxi - mini)) + new_min
        return new_data


    @staticmethod
    def get_entropy(X):
        """
        :param X: Input array 
        :return: Entropy of the input array
        """
        probs = [np.mean(X == c) for c in set(X)]
        return np.sum(-p * np.log2(p) for p in probs)


    @staticmethod
    def denoise_by_wavelets(audio,wavelet = 'dmey',threshold = 9):
        """
        Audio denoising by using wavelet packet decomposition
        Steps 1) Wavelet Packet decomposition 2) Thresholding 3) Reconstruction of wavelet packet decomposition.
        :param audio: 
        :param wavelet: 
        :param threshold: Threshold used to remove noise (Actual threshold = threshold*std of 
                            lowest level detail coefficients of the tree of wavelet packet decomposition)
        :return: Denoised audio
        """
        wp = pywt.WaveletPacket(data=audio, wavelet=wavelet, mode='symmetric')
        new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='symmetric')

        ld = wp['d'].data
        threshold = threshold*np.std(ld)

        print("Denoising using wavelets for {} levels ... This may take a while".format(wp.maxlevel))

        for i in range(wp.maxlevel):
            paths = [node.path for node in wp.get_level(i+1, 'natural')]

            for path in paths:
                new_wp[path] = pywt.threshold(wp[path].data,threshold)

        new_wp.reconstruct(update=True)
        return new_wp.data

    @staticmethod
    def get_stft(data,n_fft,win_length,hop_length):
        """
        Compute Short Time Fourier Transform of the audio 
        :param data: audio data
        :param n_fft: FFT length
        :param win_length: Time frame or the window length
        :param hop_length: Hop length between the time frames. (Determines overlapping between frames)
        :return: STFT of the audio signal
        """
        stft = librosa.stft(y = data,n_fft=n_fft,hop_length=hop_length,win_length=win_length)
        return stft

    @staticmethod
    def get_energy(data,frame_length,hop_length):
        """
        Compute the Root mean square energy of the signal 
        :param data: audio data
        :param frame_length: window or frame legth 
        :param hop_length: overlapping factor
        :return: Energy of the audio signal.
        """
        energy = librosa.feature.rmse(y=data,n_fft=frame_length,hop_length=hop_length)
        energy = energy[0,:]
        return energy


    @staticmethod
    def get_spectrogram(data,n_fft = 512,win_length = 480,hop_length = 120,range = (0,255),pixel_type = np.uint8,log_amplitude = True):

        """
        return spectorgram in log scale recaled to given range
        :param log_amplitude: if True, returns spectrogram in logamplitude, or returns linear amplitude.
        :return: Spectrogram image 
        """

        # calculating stft for window length = 480 and overlap  = 360 samples
        stft = AudioProcessing.get_stft(data,n_fft,win_length,hop_length)

        db = np.absolute(stft)

        if log_amplitude:
            db = librosa.logamplitude(db)

        # converting to log amplitude and rescaling it between the given range
        db = AudioProcessing.rescaleAmplitude(db,range)
        db = db.astype(pixel_type)

        return db

    @staticmethod
    def get_spectrogram_label(data,n_fft = 512,win_length = 480,hop_length = 120,
                              range = (0,255),pixel_type = np.uint8,log_amplitude = True,
                              initial_labels = [25,50,75,100,125,150,175,200,225,250], no_labels = 2 ):
        """
        Performs preprocessing and clustering on the spectrogram to retrieve the most prominent parts as labels.
        :param data: audio data
        :param n_fft: FFT length
        :param win_length: Window length
        :param hop_length: Hop length (overlapping factor)
        :param range: range of the intensity values of spectrogram
        :param pixel_type: Pixel type for intensity values of spectrogram
        :param log_amplitude: Whether to consider log amplitude of spectrogram or not 
        :param initial_labels: Initial Labels for clustering the spectrogram using Kmeans 
        :param no_labels: Maximum number of labels to be retained.
        :return: Labels extracted from spectrogram.
        """

        # obtaining the spectrogram of the audio
        spectrogram = AudioProcessing.get_spectrogram(data,n_fft=n_fft,win_length=win_length,hop_length=hop_length,range=range,pixel_type=pixel_type,log_amplitude = log_amplitude)

        # converting to sitk image
        db_sitk = sitk.GetImageFromArray(spectrogram)

        db_sitk = sitk.GetImageFromArray(ImageProcessing.median_image_filter(db_sitk,radius=(3,3,3)))

        # kmeans clustering the image acoording to the intial labels
        labels = sitk.ScalarImageKmeans(db_sitk,initial_labels,True)

        # considering only last n labels given byu no_labels
        lables_arr = sitk.GetArrayFromImage(labels)
        max_label = np.max(lables_arr)
        lables_arr[lables_arr < (max_label-(no_labels - 1))] = 0
        lables_arr[lables_arr >= (max_label-(no_labels - 1))] = 1

        labels = sitk.GetImageFromArray(lables_arr)

        # performing binary closing and dilating with certain parameters
        closed = sitk.BinaryMorphologicalClosing(labels,1,sitk.sitkBall)
        dilated = sitk.BinaryDilate(closed,3,sitk.sitkBall)

        # filling holes
        holesfilled = sitk.BinaryFillhole(dilated,fullyConnected=True)

        # getting the connected components and relabelling it according to size
        connected = sitk.ConnectedComponent(holesfilled,fullyConnected=True)
        relabelled = sitk.RelabelComponent(connected,minimumObjectSize=200)
        relabelled_arr = sitk.GetArrayFromImage(relabelled)

        # returning the spectrogram and the label
        return relabelled_arr

    @staticmethod
    def segmentAudioByEnergyApproximation(data,fs,threshold = 5 ,short_energy_time = 64,max_segments = 5):

        """
        Segmenting the audio based on approximation using signal energy. Modelling the noise
            by considering certain amount of low energy level frames. 
        :param data: 
        :param fs: 
        :param threshold: 
        :param short_energy_time: 
        :param max_segments: 
        :return: 
        """

        total_samples = 0.2*fs
        min_energy_samples = np.sort(abs(data))[:int(total_samples)]

        min_energy_samples = np.array(min_energy_samples)

        mean = np.mean(abs(min_energy_samples))
        std = np.std(abs(min_energy_samples))

        if std == 0.0:
            std = 0.01


        # Approximating a frame with the maximum value of the frame to eliminate the high frequency content
        approximate = np.copy(abs(data))

        i = 0
        hop_size = 2048

        while(i < data.size):
            if(i+hop_size < data.size):
                # approximate my maximum
                approximate[i:i+hop_size] = np.max(approximate[i:i+hop_size])
            else:
                approximate[i:] = np.max(approximate[i:])
            i = i+hop_size


        check_array = (abs(approximate) - mean)/float(std)

        if 0:

            import pdb
            pdb.set_trace()

            plt.plot(check_array)
            plt.show()

        if np.min(check_array )> threshold:
            threshold = np.min(check_array) + 3

        ind_p = np.where(check_array > threshold)
        ind_n = np.where(check_array <= threshold)

        check_array[ind_p] = 1
        check_array[ind_n] = 0


        diff = np.ediff1d(check_array)

        ones = np.where(diff == 1)[0]
        minus_ones = np.where(diff == -1)[0]

        if ones.size == 0:
            ones = np.array([0])

        if minus_ones.size == 0:
            minus_ones = np.array([check_array.size - 1])

        if ones[0] >= minus_ones[0]:
            ones = np.append(0,ones)

        if ones[-1] >= minus_ones[-1]:
            minus_ones = np.append(minus_ones,[check_array.size - 1])

        segments = []

        if 0:

            import pdb
            pdb.set_trace()

        for i in range(ones.size):
            if(minus_ones[i] - ones[i] >= 6144):
                # print(minus_ones[i] - ones[i],i)
                segments.append((ones[i],minus_ones[i],minus_ones[i]-ones[i]))

        def seg_size(x):
            return (x[2])

        segments = sorted(segments,key=seg_size,reverse=True)

        if len(segments) > max_segments :
            segments =segments[:5]

        return segments


    @staticmethod
    def segmentAudioBySpectrograms(data,spec_label,win_len,hop_len,max_segments = 5):
        """
        Segmentation audio by using labels generated by spectrogram.
        First compute spectrogram labels using get_spectrogram_label method and 
        :param data: audio data to be segmented 
        :param spec_label: Spectrogram labels 
        :param win_len: Window length 
        :param hop_len: Hop Length 
        :param max_segments: Maximum number of segments to be retained
        :return: Segments by removing unwanted part of the signal.
        """

        shape = spec_label.shape
        time_range = shape[1]

        check_array = np.zeros(data.size)

        for i in range(time_range):
            col_value = np.sum(spec_label[:,i])
            if col_value > 0 :
                check_array[i*hop_len : (i*hop_len + win_len)] = 1

        diff = np.ediff1d(check_array)

        ones = np.where(diff == 1)[0]
        minus_ones = np.where(diff == -1)[0]

        if ones.size == 0:
            ones = np.array([0])

        if minus_ones.size == 0:
            minus_ones = np.array([check_array.size - 1])

        if ones[0] >= minus_ones[0]:
            ones = np.append(0,ones)

        if ones[-1] >= minus_ones[-1]:
            minus_ones = np.append(minus_ones,[check_array.size - 1])

        segments = []

        for i in range(ones.size):
            # print(minus_ones[i] - ones[i],i)
            segments.append((ones[i],minus_ones[i],minus_ones[i]-ones[i]))

        def seg_size(x):
            return (x[2])

        segments = sorted(segments,key=seg_size,reverse=True)

        if len(segments) > max_segments :
            segments =segments[:max_segments]

        if 0:

            ch = np.zeros(data.size)
            ch[segments[0][0]:segments[0][1]] = 1

            import matplotlib.pyplot as plt
            plt.plot(data)
            plt.plot(ch)
            plt.show()

        return segments



    @staticmethod
    def butter_lowpass_filter(data, cutoff, fs, order=5):
        """
        Low pass filter using butterworth coefficients
        """
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b,a = butter(order, normal_cutoff, btype='low', analog=False)
        y = lfilter(b, a, data)
        return y

    @staticmethod
    def butter_highpass_filter(data, cutoff, fs, order=5):
        """
        High pass filter using butterworth coefficients
        """
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b,a = butter(order, normal_cutoff, btype='high', analog=False)
        y = lfilter(b, a, data)
        return y

    @staticmethod
    def meanImage(image_arr,radius):
        """
        Blur image with MeanImageFilter
        :param image_arr: Image array 
        :param radius: radius of the kernel
        :return: Mean Image 
        """
        meanImageFilter = sitk.MeanImageFilter()
        meanImageFilter.SetRadius(radius)
        return sitk.GetArrayFromImage(meanImageFilter.Execute(sitk.GetImageFromArray(image_arr)))


    @staticmethod
    def segmentationByIterativeTimeDomain(data,fs):
        data_copy = np.copy(data)
        energy = AudioProcessing.get_energy(data_copy,frame_length=64,hop_length=64)
        pre_threshold = None
        annotation = np.ones(energy.size)

        while 1:

            check_indices = np.where(annotation == 1)
            db = 10*np.log10(energy[check_indices])

            # db[np.isneginf(db)] = 0
            # nonzero = db[np.nonzero(db)]
            min_energy_sample = sorted(db)[0]

            print(min_energy_sample)

            threshold = 0.5*(10**((min_energy_sample)/10.0))

            if pre_threshold is not None:
                print(pre_threshold - threshold)

            pre_threshold = threshold

            data_copy[abs(data_copy) < threshold] = 0

            plt.plot(data)
            plt.plot(data_copy)
            plt.show()

            import pdb
            pdb.set_trace()


    @staticmethod
    def get_hilbert_transform(data):
        from scipy.signal import hilbert
        return hilbert(data)

    @staticmethod
    def get_audio_features(y,sr,n_fft,hop_length,n_mfcc):

        """
        Compute acoustic features of the audio 
        :param y: audio data 
        :param sr: Sampling rate 
        :param n_fft: FFT length    
        :param hop_length: Hop length 
        :param n_mfcc: Number of MFCC coefficients. 
        :return: Audio feature matrix 
        """

        features = None

        #MFCCS
        mfccs =  librosa.feature.mfcc(y=y, sr=sr, n_mfcc = n_mfcc , n_fft = n_fft, hop_length = hop_length)
        features = mfccs

        #Delta mfccs
        delta_mfccs =  librosa.feature.delta(mfccs)
        features = np.concatenate((features,delta_mfccs))


        #rmse
        rmse =  librosa.feature.rmse(y=y , n_fft = n_fft , hop_length = hop_length)
        features = np.concatenate((features,rmse))


        #spectral centroid
        spectral_centroid =  librosa.feature.spectral_centroid(y=y, sr=sr, n_fft = n_fft, hop_length = hop_length )
        features = np.concatenate((features,spectral_centroid))


        #spectral bandwidth
        spectral_bandwidth =  librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft = n_fft, hop_length = hop_length)
        features = np.concatenate((features,spectral_bandwidth))


        #spectral contrast
        spectral_contrast =  librosa.feature.spectral_contrast(y=y, sr=sr, n_fft = n_fft, hop_length = hop_length)
        features = np.concatenate((features,spectral_contrast))


        #spectral rolloff
        spectral_rolloff =  librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft = n_fft, hop_length = hop_length)
        features = np.concatenate((features,spectral_rolloff))



        #zero crossing rate
        zero_crossing_rate =  librosa.feature.zero_crossing_rate(y=y, frame_length = n_fft, hop_length = hop_length)
        features = np.concatenate((features,zero_crossing_rate))

        #
        # window = np.hanning(n_fft)
        #
        # i = 0
        # lpc_coefficients = []
        # while i <= y.shape[0]:
        #
        #     window_end = i + n_fft
        #     audio_end = y.shape[0]
        #
        #     if audio_end - i < n_fft:
        #         d = y[i:]
        #         d_len = len(d)
        #         diff = n_fft - d_len
        #
        #         d = list(d)
        #         for j in range(diff):
        #             d.append(0)
        #
        #         d = np.array(d)
        #         d = d*window
        #     else:
        #         d = y[i:window_end]
        #         d = np.array(d)
        #         d = d*window
        #
        #     lpcs,e,k = lpc(d,25)
        #     lpcs = lpcs[1:]
        #     lpc_coefficients.append(lpcs)
        #     i = i + hop_length
        #
        # lpc_coefficients = np.array(lpc_coefficients)
        #
        #
        # lpc_coefficients = np.transpose(np.array(lpc_coefficients))
        # lpc_coefficients = lpc_coefficients.astype(np.float64)


        # features = np.concatenate((features,lpc_coefficients))


        return np.transpose(features)

    @staticmethod
    def levinson_1d(r, order):

        try:
            nonzero = np.nonzero(r)[0][0]
        except:
            import pdb
            pdb.set_trace()

        r = r[nonzero:]

        r = np.atleast_1d(r)
        if r.ndim > 1:
            raise ValueError("Only rank 1 are supported for now.")

        n = r.size
        if order > n - 1:
            raise ValueError("Order should be <= size-1")
        elif n < 1:
            raise ValueError("Cannot operate on empty array !")

        if not np.isreal(r[0]):
            raise ValueError("First item of input must be real.")
        elif not np.isfinite(1/r[0]):
            raise ValueError("First item should be != 0")

        # Estimated coefficients
        a = np.empty(order+1, r.dtype)
        # temporary array
        t = np.empty(order+1, r.dtype)
        # Reflection coefficients
        k = np.empty(order, r.dtype)

        a[0] = 1.
        e = r[0]

        for i in xrange(1, order+1):
            acc = r[i]
            for j in range(1, i):
                acc += a[j] * r[i-j]
            k[i-1] = -acc / e
            a[i] = k[i-1]

            for j in range(order):
                t[j] = a[j]

            for j in range(1, i):
                a[j] += k[i-1] * np.conj(t[i-j])

            e *= 1 - k[i-1] * np.conj(k[i-1])

        return a, e, k


    @staticmethod
    def get_lpc_coefficients_feature_vector(y,order,n_fft,hop_length):

        window = np.hanning(n_fft)

        i = 0
        lpc_coefficients = []
        while i <= y.shape[0]:

            window_end = i + n_fft
            audio_end = y.shape[0]

            if audio_end - i < n_fft:
                d = y[i:]
                d_len = len(d)
                diff = n_fft - d_len

                d = list(d)
                for j in range(diff):
                    d.append(0)

                d = np.array(d)
                d = d*window
            else:
                d = y[i:window_end]
                d = np.array(d)
                d = d*window


            if not np.all(d == 0):
                a,e,k = AudioProcessing.levinson_1d(d,order)
                a = a[1:]

                if np.nan not in a and np.nan not in k:
                    lpcs = []
                    lpcs.extend(a)
                    lpcs.extend(k)
                    lpc_coefficients.append(lpcs)

            i = i + hop_length

        lpc_coefficients = np.array(lpc_coefficients)
        return lpc_coefficients


    @staticmethod
    def get_lpc_column_names(order):
        a = []
        k = []
        for i in range(order):
            a.append("LPC_A_{}".format(i+1))
            k.append("LPC_K_{}".format(i+1))

        lpc_columns = []
        lpc_columns.extend(a)
        lpc_columns.extend(k)

        return lpc_columns


    @staticmethod
    def get_audio_feature_columns(n_mfcc,append = None):

        cols = []
        mfccs = []
        delta_mfccs = []
        constrasts = []
        for i in range(n_mfcc):
            mfccs.append('MFCC_{}'.format(i+1))
            delta_mfccs.append('DELTA_MFCC_{}'.format(i+1))

        for i in range(7):
            constrasts.append('SpectralContrast_{}'.format(i+1))

        cols.extend(mfccs)
        cols.extend(delta_mfccs)
        cols.extend(['RMSE','SpectralCentroid','SpectralBandwidth'])
        cols.extend(constrasts)
        cols.extend(['SpectralRollOff','ZeroCrossingRate'])

        new_cols = []
        if append is not None:
            for col in cols:
                new_cols.append("Audio_" + col + append)

            return new_cols

        return cols
