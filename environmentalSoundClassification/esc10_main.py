from soundDataBase import EscSoundDataBase
from soundClassifier import SpectrogramClassifier
import numpy as np


db_name = 'esc10'
db_src = r'D:\projects\scratch_projects\UpworkAudioClassification\soundClassification\Data\esc10\\'
db_file_type = 'ogg'

esc10  = EscSoundDataBase(db_name=db_name,db_src=db_src,db_file_type=db_file_type)

extract_features = True
classify = True


if extract_features:

    clf = SpectrogramClassifier(esc10)
    # extract features
    features = clf.extractFeatures(fs = 24000,n_fft = 512,win_length = 480,hop_length = 120,
                                  spec_range = (0,255),spec_pixel_type = np.uint8,spec_log_amplitude = True,
                                  spec_label_range = (0,255),spec_label_pixel_type = np.uint8,spec_label_log_amplitude = True,
                                  initial_labels = [25,50,75,100,125,150,175,200,225,250], no_labels = 2 ,
                                  histogram_bins = np.arange(256),histogram_density = True,
                                  glcm_distances = [3,5], glcm_angles = [0,np.pi/4.0,np.pi/2.0, 3*np.pi/4.0],n_mfcc=60,lpc_order=25,
                                  max_segments_per_file=1)

    clf.save_features('esc_10_main.xlsx')

if classify:
    # Using two clasiifiers for classification
    clf = SpectrogramClassifier(esc10)

    # getting mean accuracy for traiing without clusters
    accuracy = []

    clf_list = [r'Features\esc_10_firstorder_glcm.xlsx']


    probabilities = np.zeros((80,10,len(clf_list)))
    target_names = ['Dog bark','Rain','Sea waves','Baby Cry','Clock tick',
                    'Person sneeze','Helicopter','Chainsaw','Rooster','Fire Crackling']
    for i in range(1,6):
        actual_labels = None
        for j in range(len(clf_list)):
            probs,labels =  clf.train(clf_list[j],validationNo=i)
            actual_labels = labels
            probabilities[:,:,j] = probs

        final_labels = []

        for k in range(probabilities.shape[0]):
            maximums = []
            for l in range(len(clf_list)):
                maximums.append(np.max(probabilities[k,:,l]))

            max_clf = np.argmax(maximums)

            max_clf_probs = probabilities[k,:,max_clf]
            final_label = np.argmax(max_clf_probs)

            final_label = final_label + 1
            final_labels.append(final_label)

        from sklearn.metrics import accuracy_score
        acc = accuracy_score(actual_labels,final_labels)

        from sklearn.metrics import classification_report
        print(classification_report(actual_labels,final_labels,target_names=target_names))

        import pdb
        pdb.set_trace()

        accuracy.append(acc)
        print(acc)
    print("Mean Accuracy : {}".format(np.mean(accuracy)))
