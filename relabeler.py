import numpy as np
import math
import random

def relabel(classifiers_map,base_classifiers,train_features,train_labels):
    probs = []
    for i in range(len(base_classifiers)):
        probs.append(list(classifiers_map[base_classifiers[i]].predict_proba(train_features)[:,1]))

    probabilities = []
    for i in range(train_features.shape[0]):
        ls=[]
        for j in range(len(base_classifiers)):
            ls.append((probs[j][i],j))

        probabilities.append(ls)

    for i in range(len(probabilities)):
        probabilities[i].sort()

    ranks = np.zeros((train_features.shape[0],len(base_classifiers)),dtype=np.int)
    for i in range(train_features.shape[0]):
        a=1
        for j in range(len(base_classifiers)):
            ranks[i,(probabilities[i][j])[1]] = a
            a = a + 1

    print "\nRank Matrix:"
    print ranks

    train_re_labels = []
    for i in range(train_labels.shape[0]):
        arr = ranks[i]
        if train_labels[i]==1:
            train_re_labels.append(base_classifiers[(np.where(arr==len(base_classifiers)))[0][0]])
        elif train_labels[i]==0:
            train_re_labels.append(base_classifiers[(np.where(arr==1))[0][0]])

    return train_re_labels
