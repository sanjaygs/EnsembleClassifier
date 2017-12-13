import numpy as np
import math
import random
import xgboost as xgb
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
import manipulate_dataset as mp
import base_model_selector as bms
import meta_model_trainer as mmt
import relabeler

if __name__== '__main__':
    no_of_classifiers = 5
    k = 4
    '''
    CHANGE no_of_classifiers AND k WHENEVER APPLICABLE
    '''
    dataset,labels = mp.add_features()
    print list(labels).count(1)
    print list(labels).count(0)
    #print dataset.shape
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []
    sss = StratifiedShuffleSplit(n_splits=1,test_size=0.2)
    for train,test in sss.split(dataset,labels):
        for itr1 in range(train.shape[0]):
            train_features.append(dataset[train[itr1],:])
            train_labels.append(labels[train[itr1]])
        for itr2 in range(test.shape[0]):
            test_features.append(dataset[test[itr2],:])
            test_labels.append(labels[test[itr2]])
    #a = dataset.shape[0] - int(dataset.shape[0]/10)
    print train_labels.count(1)
    print test_labels.count(1)
    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    test_features = np.array(test_features)
    test_labels = np.array(test_labels)
    #print train_labels
    #print test_features.shape,test_labels.shape

    clf_NaiveBayes = GaussianNB()
    clf_AdaBoost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(class_weight={0:1,1:5}),n_estimators=6)
    clf_RandomForest = RandomForestClassifier(class_weight={0:1,1:8})
    clf_LogisticReg = LogisticRegression(class_weight={0:1,1:8},warm_start=True)

    clf_NaiveBayes.fit(train_features,train_labels)
    clf_AdaBoost.fit(train_features,train_labels)
    clf_RandomForest.fit(train_features,train_labels)

    #clf_GradientBooster.fit(train_features,train_labels)
    clf_GradientBooster = xgb.XGBClassifier(max_depth=15, n_estimators=1000, learning_rate=0.1, scale_pos_wt=1, seed=27).fit(train_features, train_labels)

    clf_LogisticReg.fit(train_features,train_labels)


    classifiers_map = {0:clf_NaiveBayes,1:clf_AdaBoost,2:clf_RandomForest,3:clf_GradientBooster,4:clf_LogisticReg}

    print "-----------------------------------------------------------------------------------"
    #clf_NaiveBayes.fit(train_features,train_labels)
    pred = clf_NaiveBayes.predict(test_features)
    print list(pred).count(1),list(test_labels).count(1)
    print list(pred).count(0),list(test_labels).count(0)
    print precision_recall_fscore_support(test_labels,pred,average='micro')
    print precision_recall_fscore_support(test_labels,pred,average='weighted')
    print "-----------------------------------------------------------------------------------"
    #clf_AdaBoost.fit(train_features,train_labels)
    pred = clf_AdaBoost.predict(test_features)

    print list(pred).count(1),list(test_labels).count(1)
    print list(pred).count(0),list(test_labels).count(0)
    print precision_recall_fscore_support(test_labels,pred,average='micro')
    print precision_recall_fscore_support(test_labels,pred,average='weighted')
    print "-----------------------------------------------------------------------------------"
    #clf_RandomForest.fit(train_features,train_labels)
    pred = clf_RandomForest.predict(test_features)
    print list(pred).count(1),list(test_labels).count(1)
    print list(pred).count(0),list(test_labels).count(0)
    print precision_recall_fscore_support(test_labels,pred,average='micro')
    print precision_recall_fscore_support(test_labels,pred,average='weighted')
    print "-----------------------------------------------------------------------------------"
    #clf_GradientBooster.fit(train_features,train_labels)
    pred = clf_GradientBooster.predict(test_features)
    #print clf_GradientBooster.predict_proba(test_features)
    print list(pred).count(1),list(test_labels).count(1)
    print list(pred).count(0),list(test_labels).count(0)
    print precision_recall_fscore_support(test_labels,pred,average='micro')
    print precision_recall_fscore_support(test_labels,pred,average='weighted')
    print "-----------------------------------------------------------------------------------"
    #clf_LogisticReg.fit(train_features,tr(ain_labels)
    pred = clf_LogisticReg.predict(test_features)
    print list(pred).count(1),list(test_labels).count(1)
    print list(pred).count(0),list(test_labels).count(0)
    print precision_recall_fscore_support(test_labels,pred,average='micro')
    print precision_recall_fscore_support(test_labels,pred,average='weighted')
    print "-----------------------------------------------------------------------------------"

    base_classifiers = bms.select_base_models(classifiers_map,test_features,no_of_classifiers)
    #print base_classifiers
    #base_classifiers = [1,2]
    print "Base Classifiers : {}".format(base_classifiers)
    train_re_labels = relabeler.relabel(classifiers_map,base_classifiers,train_features,train_labels)
    #print train_re_labels
    predicted_classifiers = mmt.train_meta(classifiers_map,train_features,train_re_labels,base_classifiers,test_features,test_labels)
    final_labels = []
    correct = 0
    for i in range(test_features.shape[0]):
        #print predicted_classifiers[i]
        final_labels.append((classifiers_map[predicted_classifiers[i]].predict(test_features[i].reshape(1,-1)))[0])
        if final_labels[i] == test_labels[i]:
            correct+=1
    print final_labels.count(1),list(test_labels).count(1)
    print final_labels.count(0),list(test_labels).count(0)
    #print float(correct)/len(final_labels)
    print precision_recall_fscore_support(test_labels,final_labels,average='micro')
    print precision_recall_fscore_support(test_labels,pred,average='weighted')
