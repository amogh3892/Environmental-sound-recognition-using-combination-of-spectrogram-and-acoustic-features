from imageProcessingUtil import ImageProcessing
import sys
import numpy as np
import pandas as pd
from audio import Audio
from audioProcessingUtil import AudioProcessing
from time import time
import librosa

def svm_predict(training_samples, training_labels, test_samples, test_lables):
    from sklearn.svm import SVC
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import GridSearchCV

    parameters = {'kernel':('linear','rbf','poly','sigmoid'), 'C':(1, 10,100,1000,10000)}

    # clf = GridSearchCV(SVC(probability=True), parameters)
    # clf.fit(training_samples,training_labels)
    # pred = clf.predict_proba(test_samples)
    # return pred

    clf = GridSearchCV(SVC(probability=False), parameters)
    clf.fit(training_samples,training_labels)
    pred = clf.predict(test_samples)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(test_lables,pred)

    with open("result_labels.csv","wb") as outfile:
        np.savetxt(outfile,pred)

    return acc


def svm_best_parameters(X,y,Cs  = [0.001,0.01,0.1,1,10,100,1000,10000,100000], kernels = ['rbf','linear','poly','sigmoid'], nfolds = 5):
    from sklearn.model_selection import GridSearchCV
    from sklearn import svm

    param_grid = {'C': Cs, 'kernel' : kernels}
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    return grid_search.best_params_



class SoundClassifier(object):
    def __init__(self,database,featuresPath = None):
        self.database = database
        self.filenames = None
        self.features = None
        self.cluster_features = None
        self.cluster_normalizer = None
        self.normalized_data = None
        self.cluster_model = None
        self.labels  = None
        self.bag_of_features = None
        self.feature_columns = None

        if featuresPath is not None:
            features = pd.read_excel(featuresPath)
            self.features = features


class SpectrogramClassifier(SoundClassifier):
    def __init__(self,database,featuresPath = None):
        super(SpectrogramClassifier,self).__init__(database,featuresPath)

        self.feature_columns = ['FileName','ClassLabel','SegmentLabel']

    def extractFeatures(self,fs = 24000,n_fft = 512,win_length = 480,hop_length = 120,
                              spec_range = (0,255),spec_pixel_type = np.uint8,spec_log_amplitude = True,
                              spec_label_range = (0,255),spec_label_pixel_type = np.uint8,spec_label_log_amplitude = True,
                              initial_labels = [25,50,75,100,125,150,175,200,225,250], no_labels = 2 ,
                              histogram_bins = np.arange(256),histogram_density = True,
                              glcm_distances = [1,3,5], glcm_angles = [0,np.pi/4.0,np.pi/2.0, 3*np.pi/4.0],
                              n_mfcc = 60,lpc_order = 25,max_segments_per_file = 5 ):
        features = None

        # iterating through all categories
        for cls in self.database.classes:
            print("Class : {}".format(cls))

            # iterating through files of one particular category
            for file in self.database.get_files(cls):
                print("File : {}".format(file))

                # resampling segment to 24000 khz
                audio = Audio(file,sampling_rate=fs)

                # removing low frequence components below 500Hz
                data = AudioProcessing.butter_highpass_filter(audio.data,500,fs)

                # Applying k-means and obtianing labels from the spectrogram
                spec_labels = AudioProcessing.get_spectrogram_label(data,n_fft=n_fft,
                                                          win_length=win_length,
                                                          hop_length=hop_length,
                                                          range=spec_label_range,
                                                          pixel_type = spec_label_pixel_type,
                                                          log_amplitude=spec_label_log_amplitude,
                                                          initial_labels=initial_labels,
                                                          no_labels=no_labels)

                # obtaining segments from audio file using the labels obtained from spectrogram segmentation
                segments = AudioProcessing.segmentAudioBySpectrograms(data,spec_labels,win_length,hop_length,max_segments=max_segments_per_file)

                # extracting different features
                print("Extracting features..")
                i = 0

                # iterating through all segments to obtain features
                for segment in segments:
                    seg_data = data[segment[0]:segment[1]]

                    # obtaining spectrogram with higher resolution of the segment
                    seg_spec = AudioProcessing.get_spectrogram(seg_data,n_fft=2*n_fft,win_length=win_length,hop_length=int(hop_length/2),range=spec_range,pixel_type = spec_pixel_type,log_amplitude=spec_log_amplitude)

                    # median filtering with radius - 3
                    med = ImageProcessing.median_image_filter(seg_spec,radius=(3,3,3))

                    general_info = np.column_stack([file,cls,i])

                    # first order statistics of the spectrogram image
                    pixelFeatures = ImageProcessing.getPixelFeatureVector(med,histogram_bins=histogram_bins,histogram_density = histogram_density)

                    # extracting acoustic features
                    audio_features = AudioProcessing.get_audio_features(seg_data,fs,n_fft,hop_length,n_mfcc)
                    audio_features_mean = np.mean(audio_features,axis=0)
                    audio_features_mean = np.column_stack(audio_features_mean)
                    audio_features_var = np.var(audio_features,axis=0)
                    audio_features_var = np.column_stack(audio_features_var)

                    # obtaining glcm features
                    glcmFeatures = ImageProcessing.getGLCMFeatureVector(med,distances=glcm_distances,angles=glcm_angles)

                    # concatenating all the features to obtain feature vector of a segment
                    singleSegFeatures = np.concatenate((general_info,pixelFeatures,audio_features_mean,audio_features_var,glcmFeatures),axis = 1)


                    if features is None:
                        features = singleSegFeatures

                    else:
                        features = np.concatenate((features,singleSegFeatures))

                    i = i + 1


        # forming header columns for the dataframe
        pixelColumns = ImageProcessing.getPixelFeatureVectorColumns()

        glcmColumns = ImageProcessing.getGLCMColumnNames(distances=glcm_distances,angles=glcm_angles)

        self.feature_columns.extend(pixelColumns)

        audio_feature_mean_columns = AudioProcessing.get_audio_feature_columns(n_mfcc,'mean')
        audio_feature_var_columns = AudioProcessing.get_audio_feature_columns(n_mfcc,'var')
        self.feature_columns.extend(audio_feature_mean_columns)
        self.feature_columns.extend(audio_feature_var_columns)

        self.feature_columns.extend(glcmColumns)
        self.features = features

        return features

    def save_features(self,featuresPath):
        df = pd.DataFrame(self.features,columns= self.feature_columns)
        df.to_excel(featuresPath)


    def train(self,featuresPath,validationNo = 1):
        df = pd.read_excel(featuresPath)

        # if self.database.db_name == 'esc10' or self.database.db_name == 'esc50':
        #     df = df.drop('Maximum',axis = 1)
        #     df = df.drop('Range',axis = 1)
        #     df = df.drop('SegmentLabel',axis = 1)

        # if self.database.db_name == 'freiburg':


        try:

            df = df.drop('FirstOrder_Maximum_0',axis = 1)
            df = df.drop('FirstOrder_Range_0',axis = 1)

            df = df.drop('FirstOrder_Maximum_1',axis = 1)
            df = df.drop('FirstOrder_Range_1',axis = 1)

            df = df.drop('FirstOrder_Maximum_2',axis = 1)
            df = df.drop('FirstOrder_Range_2',axis = 1)

            df = df.drop('FirstOrder_Maximum_3',axis = 1)
            df = df.drop('FirstOrder_Range_3',axis = 1)

        except:
            pass

        df = df.drop('SegmentLabel',axis = 1)

        # general = df.filter(items=['FileName','ClassLabel'])
        # glcm = df.filter(regex='Glcm', axis=1)
        # audio = df.filter(regex='Audio',axis =1)
        # firstorder = df.filter(regex='FirstOrder',axis=1)
        #
        # general_matrix = general.as_matrix()
        # glcm_matrix = glcm.as_matrix()
        # audio_matrix = audio.as_matrix()
        # firstorder_matrix = firstorder.as_matrix()
        #
        # general_columns = general.columns.values.tolist()
        # glcm_columns = glcm.columns.values.tolist()
        # audio_columns = audio.columns.values.tolist()
        # firstorder_columns = firstorder.columns.values.tolist()
        #
        # general_columns.extend(glcm_columns)
        #
        # df = pd.DataFrame(np.concatenate((general_matrix,glcm_matrix),axis=1),columns=general_columns)


        validationNumbers = []
        filenames = df['FileName']


        for filename in filenames:

            if self.database.db_name == 'freiburg':
                f = filename.split('\\')[-1].replace('take','')
                f = f.replace('.wav','')
                f = int(f)

                cross_val = 10
                r = range(cross_val)

                for j in r:
                    if (f % cross_val) == j:
                        validationNumbers.append('{}'.format(j+1))

            else:
                validationNumbers.append(filename.split('\\')[-1][0])


        df['ValidationNo'] = pd.Series(validationNumbers)

        validation_groupy = df.groupby(['ValidationNo','ClassLabel'])

        features_train = None
        features_test = None
        labels_train = None
        labels_test = None


        # df = df.drop('FileName',axis=1)

        print("Dividing into cross validating set ...")
        for name,group in validation_groupy:
            l = int(name[1])

            group_df = group.drop('FileName',axis = 1)
            # group_df = group_df.drop('ClassLabel',axis = 1)
            group_df = group_df.drop('ValidationNo',axis = 1)

            g = group_df.as_matrix()

            f = g[:,1:]
            l = g[:,0]

            if validationNo == int(name[0]):
                if features_test is None:
                    features_test = f
                    labels_test = l
                    # labels_test.append(l)
                else:
                    features_test = np.concatenate((features_test,f))
                    labels_test = np.concatenate((labels_test,l))
                    # labels_test.append(l)

            else:
                if features_train is None:
                    features_train = f
                    labels_train = l
                    # labels_train.append(l)
                else:
                    features_train = np.concatenate((features_train,f))
                    labels_train = np.concatenate((labels_train,l))
                    # labels_train.append(l)



        # print(labels_train.shape)
        # print(labels_test.shape)


        # X = np.concatenate((features_train,features_test),axis=0)
        # y = np.concatenate((labels_train,labels_test),axis=0)


        from sklearn.preprocessing import MinMaxScaler
        from sklearn.preprocessing import StandardScaler
        normalizer = MinMaxScaler()
        # normalizer = StandardScaler()
        normalizer.fit(features_train)
        features_train = normalizer.transform(features_train)
        features_test = normalizer.transform(features_test)

        # try:
        #     from sklearn.decomposition import PCA
        #     pca = PCA(n_components=250)
        #     features_train = pca.fit_transform(features_train)
        #     features_test = pca.transform(features_test)
        # except:
        #     pass


        # print(features_train.shape)
        print("predicting data...")
        # return svm_predict(features_train,labels_train,features_test,labels_test)
        return svm_predict(features_train,labels_train,features_test,labels_test),labels_test

        # # third best till now randomforest, n_estimators, entropy 1000 estimators. 86.5 accuracy !
        # second best till now svm sigmoid C = 10000  86.75
        # third best till now svm linear C = 10000 with pca of 200 acc - 87

        # 88 using grid search svm and pca = 200

        # frieburg 200 pca, 97.25
        # frieburg 250 97.52
