import SimpleITK as sitk
import numpy as np
import scipy.stats as stats
import pandas as pd
from skimage.feature import greycomatrix, greycoprops

class ImageProcessing(object):

    def __init__(self):
        pass

    @staticmethod
    def rescaleAmplitude(image,scale_range = (0,1)):
        mini = np.min(image)
        maxi = np.max(image)

        new_min = scale_range[0]
        new_max = scale_range[1]

        new_data = ((new_max - new_min)*(image - mini)/(maxi - mini)) + new_min
        # print("Old min-max :{}-{}, New min-max: {}:{}".format(mini,maxi,new_min,new_max))
        return new_data

    @staticmethod
    def getEnergy(image):
        return np.sum(np.square(image))

    @staticmethod
    def getEntropy(arr):

        lg = np.log2(arr)
        lg[np.isneginf(lg)] = 0

        return np.sum(-1.0*arr*lg)


    @staticmethod
    def get10Percentile(arr):
        return (np.percentile(arr, 10))

    @staticmethod
    def get90Percentile(arr):
        return (np.percentile(arr, 90))

    @staticmethod
    def getInterquartileRange(arr):

        r"""
        Calculate the interquartile range of the image array.

        :math:`interquartile\ range = \textbf{P}_{75} - \textbf{P}_{25}`, where :math:`\textbf{P}_{25}` and
        :math:`\textbf{P}_{75}` are the 25\ :sup:`th` and 75\ :sup:`th` percentile of the image array, respectively.
        """

        return np.percentile(arr, 75) - np.percentile(arr, 25)

    @staticmethod
    def getMeanAbsoluteDeviation(arr):
        r"""
        Calculate the Mean Absolute Deviation for the image array.

        :math:`mean\ absolute\ deviation = \frac{1}{N}\displaystyle\sum^{N}_{i=1}{|\textbf{X}(i)-\bar{X}|}`

        Mean Absolute Deviation is the mean distance of all intensity values from the Mean Value of the image array.
        """

        return (np.mean(np.absolute((np.mean(arr) - arr))))

    @staticmethod
    def getRobustMeanAbsoluteDeviation(arr):
        r"""
        Calculate the Robust Mean Absolute Deviation for the image array.

        :math:`robust\ mean\ absolute\ deviation = \frac{1}{N_{10-90}}\displaystyle\sum^{N_{10-90}}_{i=1}{|\textbf{X}_{10-90}(i)-\bar{X}_{10-90}|}`

        Robust Mean Absolute Deviation is the mean distance of all intensity values
        from the Mean Value calculated on the subset of image array with gray levels in between, or equal
        to the 10\ :sup:`th` and 90\ :sup:`th` percentile.
        """

        prcnt10 = ImageProcessing.get10Percentile(arr)
        prcnt90 = ImageProcessing.get90Percentile(arr)
        percentileArray = arr[(arr >= prcnt10) * (arr <= prcnt90)]

        return np.mean(np.absolute(percentileArray - np.mean(percentileArray)))

    @staticmethod
    def getRootMeanSquareError(arr):
        return (np.sqrt((np.sum(arr ** 2)) / float(arr.size)))


    @staticmethod
    def getRange(arr):
        return np.max(arr) - np.min(arr)

    @staticmethod
    def getArrayHistogram(arr,bins=np.arange(256),density = True):

        hist,bins =  np.histogram(arr,bins=bins)
        if density == True:
            hist = hist/float(np.sum(hist))

        return hist,bins


    @staticmethod
    def getShapeFeatures(labelled_image):

        """
        :param labelled_image: Image with different labels for which the features are to be extracted.
                (sitk image or numpy array)
        :return:
        """

        if isinstance(labelled_image,(np.ndarray)):
            labelled_image = sitk.GetImageFromArray(labelled_image)

        labelShapeStatisticsImageFilter = sitk.LabelShapeStatisticsImageFilter()
        labelShapeStatisticsImageFilter.Execute(labelled_image)


        labelled_image = sitk.GetArrayFromImage(labelled_image)
        max_label = np.max(labelled_image)

        labels = labelShapeStatisticsImageFilter.GetLabels()

        centroidy = []
        roundness = []
        flatness = []


        for i in range(1,max_label+1):
            centroid = labelShapeStatisticsImageFilter.GetCentroid(i)
            centroidy.append(centroid[0])
            roundness.append(labelShapeStatisticsImageFilter.GetRoundness(i))
            flatness.append(labelShapeStatisticsImageFilter.GetFlatness(i))

        shapeFeatures = np.column_stack((centroidy,roundness,flatness))

        return shapeFeatures


    @staticmethod
    def getPixelFeatures(image,labelled_image,file_name,cls_label,histogram_bins = np.arange(256),histogram_density = True):
        """
        :param image: Original image.
        :param labelled_image: Image with different labels for which the features are to be extracted.
        (sitk image or numpy array)
        :return: a numpy matrix of pixel features
        """
        if isinstance(image,(sitk.Image)):
            image = sitk.GetArrayFromImage(image)

        if isinstance(labelled_image,(sitk.Image)):
            labelled_image = sitk.GetArrayFromImage(labelled_image)


        minimums = []
        maximums = []
        means = []
        medians = []
        variances = []
        energies = []
        entropies = []
        tenPercentiles = []
        nintyPercentiles = []
        interquartileRanges = []
        ranges = []
        meanAbsoluteDeviations = []
        robustMeanAbsoluteDeviations = []
        rootMeanSquareErrors = []
        skewness = []
        kurtosis = []
        cls_labels = []
        file_names = []

        max_label = np.max(labelled_image)

        for i in range(1,max_label+1):
            label = i
            label_indices = np.where(labelled_image == label)
            label_pixels = image[label_indices]
            minimums.append(np.min(label_pixels))
            maximums.append(np.max(label_pixels))
            means.append(np.mean(label_pixels))
            medians.append(np.median(label_pixels))
            variances.append(np.var(label_pixels))

            pdf,bins = ImageProcessing.getArrayHistogram(label_pixels,bins = histogram_bins,density = histogram_density)

            energies.append(ImageProcessing.getEnergy(pdf))
            entropies.append(ImageProcessing.getEntropy(pdf))

            tenPercentiles.append(ImageProcessing.get10Percentile(label_pixels))
            nintyPercentiles.append(ImageProcessing.get90Percentile(label_pixels))
            interquartileRanges.append(ImageProcessing.getInterquartileRange(label_pixels))
            ranges.append(ImageProcessing.getRange(label_pixels))
            meanAbsoluteDeviations.append(ImageProcessing.getMeanAbsoluteDeviation(label_pixels))
            robustMeanAbsoluteDeviations.append(ImageProcessing.getRobustMeanAbsoluteDeviation(label_pixels))
            rootMeanSquareErrors.append(ImageProcessing.getRootMeanSquareError(label_pixels))
            skewness.append(stats.skew(label_pixels))
            kurtosis.append(stats.kurtosis(label_pixels))
            cls_labels.append(cls_label)
            file_names.append(file_name)

        pixelFeatures = np.column_stack((file_names,cls_labels,minimums,maximums,means,medians,variances,energies,
                           entropies,tenPercentiles,nintyPercentiles,
                           interquartileRanges,ranges,meanAbsoluteDeviations,robustMeanAbsoluteDeviations,rootMeanSquareErrors,
                           skewness,kurtosis))
        return pixelFeatures


    @staticmethod
    def getPixelFeatureVector(image,histogram_bins = np.arange(256),histogram_density = True):
        """
        :param image: Original image.
        :param labelled_image: Image with different labels for which the features are to be extracted.
        (sitk image or numpy array)
        :return: a numpy matrix of pixel features
        """
        if isinstance(image,(sitk.Image)):
            image = sitk.GetArrayFromImage(image)


        # dividing the image into 4 parts wrt y axis
        seg_size = int(image.shape[0]/4)

        pixelFeatures = None

        for i in range(4):

            seg_image = image[int(i*seg_size):int((i+1)*seg_size),:]

            minimums = []
            maximums = []
            means = []
            medians = []
            variances = []
            energies = []
            entropies = []
            tenPercentiles = []
            nintyPercentiles = []
            interquartileRanges = []
            ranges = []
            meanAbsoluteDeviations = []
            robustMeanAbsoluteDeviations = []
            rootMeanSquareErrors = []
            skewness = []
            kurtosis = []

            label_pixels = seg_image.ravel()

            minimums.append(np.min(label_pixels))
            maximums.append(np.max(label_pixels))
            means.append(np.mean(label_pixels))
            medians.append(np.median(label_pixels))
            variances.append(np.var(label_pixels))

            pdf,bins = ImageProcessing.getArrayHistogram(label_pixels,bins = histogram_bins,density = histogram_density)

            energies.append(ImageProcessing.getEnergy(pdf))
            entropies.append(ImageProcessing.getEntropy(pdf))

            tenPercentiles.append(ImageProcessing.get10Percentile(label_pixels))
            nintyPercentiles.append(ImageProcessing.get90Percentile(label_pixels))
            interquartileRanges.append(ImageProcessing.getInterquartileRange(label_pixels))
            ranges.append(ImageProcessing.getRange(label_pixels))
            meanAbsoluteDeviations.append(ImageProcessing.getMeanAbsoluteDeviation(label_pixels))
            robustMeanAbsoluteDeviations.append(ImageProcessing.getRobustMeanAbsoluteDeviation(label_pixels))
            rootMeanSquareErrors.append(ImageProcessing.getRootMeanSquareError(label_pixels))
            skewness.append(stats.skew(label_pixels))
            kurtosis.append(stats.kurtosis(label_pixels))


            pixelFeatureVector = np.column_stack((minimums,maximums,means,medians,variances,energies,
                               entropies,tenPercentiles,nintyPercentiles,
                               interquartileRanges,ranges,meanAbsoluteDeviations,robustMeanAbsoluteDeviations,rootMeanSquareErrors,
                               skewness,kurtosis))

            if pixelFeatures is None:
                pixelFeatures = pixelFeatureVector
            else:
                pixelFeatures = np.concatenate((pixelFeatures,pixelFeatureVector),axis = 1)

        return pixelFeatures


    @staticmethod
    def getPixelFeatureVectorColumns():

        pixel_cols = []
        features = ['Minimum','Maximum','Mean','Median','Variance','Energy','Entropy','TenPentile','NintyPercentile',
                                   'InterQuartileRange','Range','MeanAbsoluteDeviation','RobustMeanAbsoluteDeviation','RootMeanSquareError',
                                   'Skewness','Kurtosis']

        for i in range(4):
            for feature in features:
                pixel_cols.append("{}_{}_{}".format("FirstOrder",feature,i))

        return pixel_cols

    @staticmethod
    def _entropy(p):
        ''' Function calcuate entropy feature'''
        p = p.ravel()
        return -np.dot(np.log2(p+(p==0)),p)

    @staticmethod
    def getGLCMFeatureVector(image,distances = [1,3,5],angles =[0,np.pi/4.0,np.pi/2.0, 3*np.pi/4.0]):
        '''
        Function calculate all co-occurence matrix based features
        Input p: gray level co-occurence matrix
        Output: List of the 13 GCLM features
        '''


        if isinstance(image,(sitk.Image)):
            image = sitk.GetArrayFromImage(image)

        feature_vector = None

        # dividing the image into 4 parts wrt y axis
        seg_size = image.shape[0]/4


        for i in range(4):

            seg_image = image[int(i*seg_size):int((i+1)*seg_size),:]

            glcm = greycomatrix(seg_image,distances, angles, 256, symmetric=True, normed=True)

            for d in range(len(distances)):
                for a in range(len(angles)):

                    p = glcm[:,:,d,a]

                    feats = np.zeros(13,np.double)
                    maxv = len(p)
                    k = np.arange(maxv)
                    k2 = k**2
                    tk = np.arange(2*maxv)
                    tk2 = tk**2
                    i,j = np.mgrid[:maxv,:maxv]
                    ij = i*j
                    i_j2_p1 = (i - j)**2
                    i_j2_p1 += 1
                    i_j2_p1 = 1. / i_j2_p1
                    i_j2_p1 = i_j2_p1.ravel()
                    px_plus_y = np.empty(2*maxv, np.double)
                    px_minus_y = np.empty(maxv, np.double)
                    pravel = p.ravel()
                    px = p.sum(0)
                    py = p.sum(1)
                    ux = np.dot(px, k)
                    uy = np.dot(py, k)
                    vx = np.dot(px, k2) - ux**2
                    vy = np.dot(py, k2) - uy**2
                    sx = np.sqrt(vx)
                    sy = np.sqrt(vy)
                    px_plus_y = np.zeros(shape=(2*p.shape[0] ))
                    px_minus_y = np.zeros(shape=(p.shape[0]))
                    for i in range(p.shape[0]):
                       for j in range(p.shape[0]):
                           p_ij = p[i,j]
                           px_plus_y[i+j] += p_ij
                           px_minus_y[np.abs(i-j)] += p_ij
                    feats[0] = np.sqrt(np.dot(pravel, pravel))                        # Energy
                    feats[1] = np.dot(k2, px_minus_y)                                 # Contrast
                    if sx == 0. or sy == 0.:
                       feats[2] = 1.
                    else:
                       feats[2] = (1. / sx / sy) * (np.dot(ij.ravel(), pravel) - ux * uy) # Correlation
                    feats[3] = vx                                                     #Sum of Squares: Variance
                    feats[4] = np.dot(i_j2_p1, pravel)                                # Inverse of Difference Moment
                    feats[5] = np.dot(tk, px_plus_y)                                  # Sum Average
                    feats[7] = ImageProcessing._entropy(px_plus_y)                                    # Sum Entropy
                    feats[6] = ((tk-feats[7])**2*px_plus_y).sum()                     # Sum Variance
                    feats[8] = ImageProcessing._entropy(pravel)                                       # Entropy
                    feats[9] = px_minus_y.var()                                       # Difference Variance
                    feats[10] = ImageProcessing._entropy(px_minus_y)                                  # Difference Entropy
                    HX = ImageProcessing._entropy(px)
                    HY = ImageProcessing._entropy(py)
                    crosspxpy = np.outer(px,py)
                    crosspxpy += (crosspxpy == 0)
                    crosspxpy = crosspxpy.ravel()
                    HXY1 = -np.dot(pravel, np.log2(crosspxpy))
                    HXY2 = ImageProcessing._entropy(crosspxpy)
                    if max(HX, HY) == 0.:
                       feats[11] = (feats[8]-HXY1)                                    # Information Measure of Correlation 1
                    else:
                       feats[11] = (feats[8]-HXY1)/max(HX,HY)
                    feats[12] = np.sqrt(max(0,1 - np.exp( -2. * (HXY2 - feats[8]))))  # Information Measure of Correlation 2

                    feats = np.column_stack(feats)
                    if feature_vector is None:
                        feature_vector = feats
                    else:
                        feature_vector = np.concatenate((feature_vector,feats),axis=1)


        return feature_vector


    @staticmethod
    def getGLCMColumnNames(distances = [1,3,5],angles =[0,np.pi/4.0,np.pi/2.0, 3*np.pi/4.0]):
        glcm_columns = []

        for i in range(4):
            for j in range(len(distances)):
                for k in range(len(angles)):
                    glcm_columns.append('Glcm_ENERGY_{}_{}_seg_{}'.format(distances[j],angles[k],i))
                    glcm_columns.append('Glcm_CONTRAST_{}_{}_seg_{}'.format(distances[j],angles[k],i))
                    glcm_columns.append('Glcm_CORRELATION_{}_{}_seg_{}'.format(distances[j],angles[k],i))
                    glcm_columns.append('Glcm_VARIANCE_{}_{}_seg_{}'.format(distances[j],angles[k],i))
                    glcm_columns.append('Glcm_INVERSE_DIFFERENCE_OF_MOMENT_{}_{}_seg_{}'.format(distances[j],angles[k],i))
                    glcm_columns.append('Glcm_SUM_AVERAGE_{}_{}_seg_{}'.format(distances[j],angles[k],i))
                    glcm_columns.append('Glcm_SUM_ENTROPY_{}_{}_seg_{}'.format(distances[j],angles[k],i))
                    glcm_columns.append('Glcm_SUM_VARIANCE_{}_{}_seg_{}'.format(distances[j],angles[k],i))
                    glcm_columns.append('Glcm_ENTROPY_{}_{}_seg_{}'.format(distances[j],angles[k],i))
                    glcm_columns.append('Glcm_DIFFERENCE_VARIANCE_{}_{}_seg_{}'.format(distances[j],angles[k],i))
                    glcm_columns.append('Glcm_DIFFERENCE_ENTROPY_{}_{}_seg_{}'.format(distances[j],angles[k],i))
                    glcm_columns.append('Glcm_INFORMATION_MEASURE_CORRELATION_1_{}_{}_seg_{}'.format(distances[j],angles[k],i))
                    glcm_columns.append('Glcm_INFORMATION_MEASURE_CORRELATION_2_{}_{}_seg_{}'.format(distances[j],angles[k],i))

        return glcm_columns


    @staticmethod
    def getPixelFeaturesHeaders():
        return ['FileName','ClassLabel','Minimum','Maximum','Mean','Median','Variance','Energy','Entropy','TenPentile','NintyPercentile',
                                   'InterQuartileRange','Range','MeanAbsoluteDeviation','RobustMeanAbsoluteDeviation','RootMeanSquareError',
                                   'Skewness','Kurtosis']




    @staticmethod
    def histogram_equalize(image,radius = (1,1,1), alpha = 0.6, beta = 0.3):

        if isinstance(image,(np.ndarray)):
            image = sitk.GetImageFromArray(image)


        filter = sitk.AdaptiveHistogramEqualizationImageFilter()

        heq = sitk.AdaptiveHistogramEqualization(image,radius,alpha = alpha, beta = beta)
        # mean = sitk.Mean(heq,(1,1))

        return sitk.GetArrayFromImage(heq)


    @staticmethod
    def median_image_filter(image,radius = (1,1,1)):
        if isinstance(image,(np.ndarray)):
            image = sitk.GetImageFromArray(image)

        med = sitk.Median(image,radius)

        return sitk.GetArrayFromImage(med)



