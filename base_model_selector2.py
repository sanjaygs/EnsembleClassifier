import numpy as np
import math
import random
import itertools
from sklearn.cluster import AgglomerativeClustering

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

    k=3
    agg = AgglomerativeClustering(n_clusters=k,affinity='euclidean',linkage='ward')
    clusters = agg.fit_predict(ranks.transpose())
    clusters = clusters.tolist()
    big_list = []
    for i in range(k):
        big_list.append([])
    for val in range(len(clusters)):
        big_list[clusters[val]].append(val)
    final_list = []
    for small_list in big_list:
        print(random.choice(small_list))
        final_list.append(random.choice(small_list))
    return final_list
