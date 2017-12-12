from soundDataBase import EscSoundDataBase

import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == "__main__":

    import pdb
    pdb.set_trace()

    db_name = 'esc10'
    db_src = r'Data\esc10\\'
    db_file_type = 'ogg'

    esc10 = EscSoundDataBase(db_name=db_name, db_src=db_src, db_file_type=db_file_type)

    from soundClassifier import SpectrogramClassifier
    clf = SpectrogramClassifier(esc10)

    # extracting and saving features
    features = clf.extractFeatures(fs=24000, n_fft=512, win_length=480, hop_length=120,
                                   spec_range=(0, 255), spec_pixel_type=np.uint8, spec_log_amplitude=True,
                                   spec_label_range=(0, 255), spec_label_pixel_type=np.uint8,
                                   spec_label_log_amplitude=True,
                                   initial_labels=[25, 50, 75, 100, 125, 150, 175, 200, 225, 250], no_labels=2,
                                   histogram_bins=np.arange(256), histogram_density=True,
                                   glcm_distances=[3, 5], glcm_angles=[0, np.pi / 4.0, np.pi / 2.0, 3 * np.pi / 4.0],
                                   n_mfcc=60, lpc_order=25,
                                   max_segments_per_file=1)

    clf.save_features('Features/esc_10_main.xlsx')

    import pdb
    pdb.set_trace()
    # classification

    # getting mean accuracy for training without clusters
    accuracy = []

    clf_list = [r'Features/esc_10_main.xlsx']


    probabilities = np.zeros((80,10,len(clf_list)))
    target_names = ['Dog bark','Rain','Sea waves','Baby Cry','Clock tick',
                    'Person sneeze','Helicopter','Chainsaw','Rooster','Fire Crackling']

    full_confusion_matrix = None

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

        # from sklearn.metrics import classification_report
        # print(classification_report(actual_labels,final_labels,target_names=target_names))

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(actual_labels, final_labels)
        if full_confusion_matrix is None:
            full_confusion_matrix = cm
        else:
            full_confusion_matrix = full_confusion_matrix + cm

        accuracy.append(acc)
        print(acc)
    print("Mean Accuracy : {}".format(np.mean(accuracy)))


    # calculating recall
    recall = []
    false_negatives = []
    true_positives = []
    for i in range(full_confusion_matrix.shape[0]):
        tp = 0
        fn = 0
        for j in range(full_confusion_matrix.shape[1]):
            if i == j:
                tp = full_confusion_matrix[i,j]
            else:
                fn = fn + full_confusion_matrix[i,j]

        r = float(tp)/(tp+fn)
        false_negatives.append(fn)
        true_positives.append(tp)
        recall.append(r)

    recall = np.array(recall)
    false_negatives = np.array(false_negatives)
    true_positives = np.array(true_positives)
    # calculating precision
    precision = []
    false_postives = []
    for j in range(full_confusion_matrix.shape[1]):
        tp = 0
        fp = 0
        for i in range(full_confusion_matrix.shape[0]):
            if i == j:
                tp = full_confusion_matrix[i, j]
            else:
                fp = fp + full_confusion_matrix[i, j]

        p = float(tp) / (tp + fp)
        precision.append(p)
        false_postives.append(fp)

    precision = np.array(precision)
    false_postives = np.array(false_postives)

    f_score = 2.0*(recall*precision)/(recall + precision)

    total_recall = float(np.sum(true_positives))/(np.sum(true_positives) + np.sum(false_negatives))
    total_precision = float(np.sum(true_positives)) / (np.sum(true_positives) + np.sum(false_postives))
    total_fscore = 2.0*(total_recall*total_precision)/(total_recall+total_precision)
    print("Recall : {}".format(recall))
    print("Precision : {}".format(precision))
    print("fscore : {}".format(f_score))
    print("Total recall : {}".format(total_recall))
    print("Total precision : {}".format(total_precision))
    print("Total fscore : {}".format(total_fscore))



    plot_confusion_matrix(full_confusion_matrix, target_names)
    plt.show()



