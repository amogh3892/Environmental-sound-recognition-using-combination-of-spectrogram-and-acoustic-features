
from soundDataBase import EscSoundDataBase
from soundClassifier import SpectrogramClassifier
import numpy as np


db_name = 'freiburg'
db_src = r'D:\projects\scratch_projects\UpworkAudioClassification\soundClassification\Data\freiburg\\'
db_file_type = 'wav'

freiburg  = EscSoundDataBase(db_name=db_name,db_src=db_src,db_file_type=db_file_type)



# if 0:
#     # extract features
#     features = clf.extractFeatures(fs = 24000,n_fft = 512,win_length = 480,hop_length = 120,
#                                   spec_range = (0,255),spec_pixel_type = np.uint8,spec_log_amplitude = True,
#                                   spec_label_range = (0,255),spec_label_pixel_type = np.uint8,spec_label_log_amplitude = True,
#                                   initial_labels = [25,50,75,100,125,150,175,200,225,250], no_labels = 2 ,
#                                   histogram_bins = np.arange(256),histogram_density = True,
#                                   glcm_distances = [3,5], glcm_angles = [0,np.pi/4.0,np.pi/2.0, 3*np.pi/4.0],n_mfcc=60,lpc_order=25,
#                                   max_segments_per_file=1)
#
#
#     clf.save_features('features__pixel_audio_glcm_4.xlsx')


if 0:

    clf = SpectrogramClassifier(freiburg)
    # extract features
    features = clf.extractFeatures(fs = 24000,n_fft = 512,win_length = 480,hop_length = 120,
                                  spec_range = (0,255),spec_pixel_type = np.uint8,spec_log_amplitude = True,
                                  spec_label_range = (0,255),spec_label_pixel_type = np.uint8,spec_label_log_amplitude = True,
                                  initial_labels = [25,50,75,100,125,150,175,200,225,250], no_labels = 2 ,
                                  histogram_bins = np.arange(256),histogram_density = True,
                                  glcm_distances = [3,5], glcm_angles = [0,np.pi/4.0,np.pi/2.0, 3*np.pi/4.0],n_mfcc=60,lpc_order=25,
                                  max_segments_per_file=1)

    clf.save_features('freiburg_main.xlsx')


if 0:
    # getting mean accuracy for traiing without clusters
    accuracy = []
    for i in range(1,11):
        acc = clf.train_without_clustering('freiburg.xlsx',validationNo=i)
        accuracy.append(acc)
        print(acc)
    print("Mean Accuracy : {}".format(np.mean(accuracy)))



if 1:
    # checking multiple classifiers
    clf = SpectrogramClassifier(freiburg)

    # getting mean accuracy for traiing without clusters
    accuracy = []

    # clf_list = [r'Features\esc_10_all.xlsx',r'Features\esc_10_audio.xlsx',
    #             r'Features\esc_10_glcm.xlsx',r'Features\esc_10_firstorder_audio.xlsx',
    #             r'Features\esc_10_firstorder_glcm.xlsx',r'Features\esc_10_audio_glcm.xlsx']

    # clf_list = [r'Features\esc_10_audio.xlsx',
    #             r'Features\esc_10_firstorder_glcm.xlsx']


    clf_list = [r'Features\freiburg_audio.xlsx',r'Features\freiburg_firstorder_glcm.xlsx']


    # probabilities = np.zeros((80,22,len(clf_list)))



    # target_names = ['Dog bark','Rain','Sea waves','Baby Cry','Clock tick',
    #                 'Person sneeze','Helicopter','Chainsaw','Rooster','Fire Crackling']

    target_names = ['Background','Food Bag Opening','Blender','Cornflakes Bowl','Cornflakes Eating',
                     'Pouring cup','Dish Washer','Electric Razor','Flatware Sorting','Food Processor',
                     'Hair Dyer','Microwave','Microwave Bell','Microwave Door','Plates Sorting',
                     'Stirring Cup','Toilet Flush','Tooth Brushing','Vacuum Cleaner','Washing Machine',
                     'Water Boiler','Water Tap']

    for i in range(1,11):
        probabilities = None
        actual_labels = None
        for j in range(len(clf_list)):
            probs,labels =  clf.train(clf_list[j],validationNo=i)
            actual_labels = labels
            if probabilities is None:
                probabilities = probs
            else:
                probabilities = np.dstack((probabilities,probs))

            # probabilities[:,:,j] = probs

        if probabilities.ndim == 2:
            probabilities = probabilities[:,:,np.newaxis]

        # c1,labels = clf.train(r'Features\esc_10_all.xlsx',validationNo=i)
        # c2,labels = clf.train(r'Features\esc_10_audio.xlsx',validationNo=i)
        # c3,labels = clf.train(r'Features\esc_10_glcm.xlsx',validationNo=i)
        # c4,labels = clf.train(r'Features\esc_10_firstorder_audio.xlsx',validationNo=i)
        # c5,labels = clf.train(r'Features\esc_10_firstorder_glcm.xlsx',validationNo=i)
        # c6,labels = clf.train(r'Features\esc_10_audio_glcm.xlsx',validationNo=i)

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

            # print final_label


        from sklearn.metrics import accuracy_score
        acc = accuracy_score(actual_labels,final_labels)

        from sklearn.metrics import classification_report
        print(classification_report(actual_labels,final_labels,target_names=target_names))


        accuracy.append(acc)
        print(acc)
    print("Mean Accuracy : {}".format(np.mean(accuracy)))



if 0:

    clf = SpectrogramClassifier(freiburg)

    # getting mean accuracy for traiing without clusters
    accuracy = []
    for i in range(1,11):
        acc = clf.train(r'Features\freiburg_firstorder_glcm.xlsx',validationNo=i)
        accuracy.append(acc)
        print(acc)
    print("Mean Accuracy : {}".format(np.mean(accuracy)))
