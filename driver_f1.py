import numpy as np
import math
import random
import xgboost as xgb
from sklearn.metrics import precision_recall_fscore_support,roc_curve,auc,zero_one_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
import manipulate_dataset as mp
import base_model_selector as bms
import base_model_selector2 as bms2
import meta_model_trainer as mmt
import relabeler
import time
from mlxtend.classifier import StackingClassifier

if __name__== '__main__':
    print "Start Time: {}".format(time.time())
    no_of_classifiers = 4
    k = 3
    dataset,labels = mp.add_features()
    print "Shape of the dataset : {}".format(dataset.shape)
    print "Number of ones in the dataset : {}".format(list(labels).count(1))
    print "Number of zeroes in the dataset : {}".format(list(labels).count(0))
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

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    zero_one_losses_list = []

    print '---------------------------------------------------------------------------------------------------------'
    print 'Bagging Classifier:'
    clf = BaggingClassifier(base_estimator=LogisticRegression(class_weight={0:1,1:9},warm_start=True),n_estimators=3)
    clf.fit(train_features,train_labels)
    pred = clf.predict(test_features)
    print "F1 Score:"
    print (precision_recall_fscore_support(test_labels,pred,average='micro'))[2]
    fpr, tpr, thresholds = roc_curve(test_labels, pred)
    print "Area Under Receiver Operating Characteristic Curve (ROC):"
    print auc(fpr, tpr)

    print '\nStacking Classifier:'
    clf = StackingClassifier(classifiers=[LogisticRegression(class_weight={0:1,1:9},warm_start=True),LogisticRegression(class_weight={0:1,1:9},warm_start=True),LogisticRegression(class_weight={0:1,1:9},warm_start=True)],meta_classifier=LogisticRegression())
    clf.fit(train_features,train_labels)
    pred = clf.predict(test_features)
    print "F1 Score:"
    print (precision_recall_fscore_support(test_labels,pred,average='micro'))[2]
    fpr, tpr, thresholds = roc_curve(test_labels, pred, pos_label=1)
    print "Area Under Receiver Operating Characteristic Curve (ROC):"
    print auc(fpr, tpr)

    print '---------------------------------------------------------------------------------------------------------'

    print "Classifiers Used : AdaBoost, RandomForest,Gradient Booster, Logistic Regression"
    print "\nTrain Start Time : {}".format(time.time())
    clf_AdaBoost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(class_weight={0:1,1:4}),n_estimators=5)
    clf_RandomForest = RandomForestClassifier(class_weight={0:1,1:11})
    clf_LogisticReg = LogisticRegression(class_weight={0:1,1:9},warm_start=True)

    clf_AdaBoost.fit(train_features,train_labels)
    clf_RandomForest.fit(train_features,train_labels)

    clf_GradientBooster = xgb.XGBClassifier(max_depth=10, n_estimators=500, learning_rate=0.1, scale_pos_wt=1, seed=27).fit(train_features, train_labels)

    clf_LogisticReg.fit(train_features,train_labels)

    print "\nClassifier Map: 0:AdaBoost 1:RandomForest 2:Gradient Booster 3.Logistic Regression"
    classifiers_map = {0:clf_AdaBoost,1:clf_RandomForest,2:clf_GradientBooster,3:clf_LogisticReg}
    print "\nTrain End Time : {}".format(time.time())

    f1_scores_list = []

    print "-----------------------------------------------------------------------------------"
    print "\nAdaBoost Classifier:"
    pred = clf_AdaBoost.predict(test_features)
    print "Number of ones predicted and number of ones in test set respectively"
    print list(pred).count(1),list(test_labels).count(1)
    print "Number of zeroes predicted and number of zeroes in test set respectively"
    print list(pred).count(0),list(test_labels).count(0)
    print "F1 Score:"
    print (precision_recall_fscore_support(test_labels,pred,average='micro'))[2]
    f1_scores_list.append((precision_recall_fscore_support(test_labels,pred,average='micro'))[2])

    print "Zero One Loss:"
    zol = zero_one_loss(test_labels,pred)
    print zol
    zero_one_losses_list.append(zol)

    fpr, tpr, thresholds = roc_curve(test_labels, pred)
    print "Area under ROC curve"
    print auc(fpr, tpr)
    print "-----------------------------------------------------------------------------------"

    print "\nRandom Forest Classifier:"
    pred = clf_RandomForest.predict(test_features)
    print "Number of ones predicted and number of ones in test set respectively"
    print list(pred).count(1),list(test_labels).count(1)
    print "Number of zeroes predicted and number of zeroes in test set respectively"
    print list(pred).count(0),list(test_labels).count(0)
    print "F1 Score:"
    print (precision_recall_fscore_support(test_labels,pred,average='micro'))[2]
    f1_scores_list.append((precision_recall_fscore_support(test_labels,pred,average='micro'))[2])

    print "Zero One Loss:"
    zol = zero_one_loss(test_labels,pred)
    print zol
    zero_one_losses_list.append(zol)

    fpr, tpr, thresholds = roc_curve(test_labels, pred)
    print "Area under ROC curve"
    print auc(fpr, tpr)
    print "-----------------------------------------------------------------------------------"
    print "\nGradient Booster Classifier:"
    pred = clf_GradientBooster.predict(test_features)
    print "Number of ones predicted and number of ones in test set respectively"
    print list(pred).count(1),list(test_labels).count(1)
    print "Number of zeroes predicted and number of ones in test set respectively"
    print list(pred).count(0),list(test_labels).count(0)
    print "F1 Score:"
    print (precision_recall_fscore_support(test_labels,pred,average='micro'))[2]
    f1_scores_list.append((precision_recall_fscore_support(test_labels,pred,average='micro'))[2])

    print "Zero One Loss:"
    zol = zero_one_loss(test_labels,pred)
    print zol
    zero_one_losses_list.append(zol)

    fpr, tpr, thresholds = roc_curve(test_labels, pred)
    print "Area under ROC curve"
    print auc(fpr, tpr)
    print "-----------------------------------------------------------------------------------"
    print "\nLogistic Regression Classifier:"
    pred = clf_LogisticReg.predict(test_features)
    print "Number of ones predicted and number of ones in test set respectively"
    print list(pred).count(1),list(test_labels).count(1)
    print "Number of zeroes predicted and number of zeroes in test set respectively"
    print list(pred).count(0),list(test_labels).count(0)
    print "F1 Score:"
    print (precision_recall_fscore_support(test_labels,pred,average='micro'))[2]
    f1_scores_list.append((precision_recall_fscore_support(test_labels,pred,average='micro'))[2])

    print "Zero One Loss:"
    zol = zero_one_loss(test_labels,pred)
    print zol
    zero_one_losses_list.append(zol)

    fpr, tpr, thresholds = roc_curve(test_labels, pred)
    print "Area under ROC curve"
    print auc(fpr, tpr)
    print "-----------------------------------------------------------------------------------"

    print "\nF1 scores:"
    print f1_scores_list
    print "\n3 Least:"
    print (sorted(f1_scores_list,reverse=True))[:3]
    indices = []
    for item in (sorted(f1_scores_list,reverse=True))[:3]:
        indices.append(f1_scores_list.index(item))

    print "\n Chosen 3 classifiers"
    print(sorted(indices))
    print "\nBase Model Selection (using highest f1 scores):\n"
    base_classifiers = sorted(indices)
    print "\nBase Classifiers : {}".format(base_classifiers)
    print "\nRelabeler:\n"
    train_re_labels = relabeler.relabel(classifiers_map,base_classifiers,train_features,train_labels)

    print "\nMeta Model Trainer:\n"
    predicted_classifiers = mmt.train_meta(classifiers_map,train_features,train_re_labels,base_classifiers,test_features,test_labels)
    print "\nMeta Model Training Finished.\n"
    final_labels = []
    correct = 0
    for i in range(test_features.shape[0]):
        final_labels.append((classifiers_map[predicted_classifiers[i]].predict(test_features[i].reshape(1,-1)))[0])
        if final_labels[i] == test_labels[i]:
            correct+=1
    print "\nPredicted and actual number of ones in final model"
    print final_labels.count(1),list(test_labels).count(1)
    print "\nPredicted and actual number of zeroes in final model"
    print final_labels.count(0),list(test_labels).count(0)
    print "\nFinal F1 Score:"
    print (precision_recall_fscore_support(test_labels,final_labels,average='micro'))[2]
    fpr, tpr, thresholds = roc_curve(test_labels, final_labels)
    print "Area under ROC"
    print auc(fpr, tpr)
    print "\nEnd Time : {}".format(time.time())
