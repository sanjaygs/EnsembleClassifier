import numpy as np
import math
import random
import itertools

def select_base_models(classifiers_map,test_features,no_of_classifiers):
    probs = []
    for i in range(no_of_classifiers):
        probs.append(list(classifiers_map[i].predict_proba(test_features)[:,1]))

    probabilities = []
    for i in range(test_features.shape[0]):
        ls=[]
        for j in range(no_of_classifiers):
            ls.append((probs[j][i],j))

        probabilities.append(ls)

    for i in range(len(probabilities)):
        probabilities[i].sort()

    ranks = np.zeros((test_features.shape[0],no_of_classifiers),dtype=np.int)
    for i in range(test_features.shape[0]):
        ls = []
        a=1
        for j in range(no_of_classifiers):
            ranks[i,(probabilities[i][j])[1]] = a
            a = a + 1
    print "\nRanks Matrix:"
    print ranks

    diversity_matrix = np.zeros((no_of_classifiers,no_of_classifiers),dtype=np.float)

    all_pairs = list(itertools.combinations(range(no_of_classifiers),2))
    rmsds = []
    for pair in all_pairs:
        sum_of_squares = 0
        mean_sos = 0
        rmsd = 0
        for i in range(ranks.shape[0]):
            sum_of_squares += math.pow(ranks[i,pair[0]]-ranks[i,pair[1]],2)
        mean_sos = float(sum_of_squares)/ranks.shape[0]
        rmsd = math.sqrt(mean_sos)
        rmsds.append(rmsd)
        diversity_matrix[pair[0],pair[1]] = rmsd
    print "\nDiversity Matrix:"
    print diversity_matrix

    rmsds.sort()
    classifiers = zip(*np.where(diversity_matrix == rmsds[0]))
    base_classifiers = []
    for i in range(no_of_classifiers):
        if i!=classifiers[0][0] and i!=classifiers[0][1]:
            base_classifiers.append(i)
    base_classifiers.append(classifiers[0][random.choice([0,1])])
    return base_classifiers
