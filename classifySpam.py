# -*- coding: utf-8 -*-
"""
Demo of XGBoost GridSearch with multiple evaluation metrics using 5-fold cross-validation on spam data
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb

desiredFPR = 0.01

def aucCV(features, labels):
    xgb_model = xgb.XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        learning_rate=0.1,
        max_depth=3,
        n_estimators=200,
        subsample=1.0
    )
    imputer = KNNImputer(missing_values=-1, n_neighbors=3)
    features_imputed = imputer.fit_transform(features)
    auc_scores = cross_val_score(xgb_model, features_imputed, labels, cv=10, scoring='roc_auc', n_jobs=-1)
    return np.mean(auc_scores)

def tprAtFPR(labels, outputs, desiredFPR):
    fpr, tpr, thres = roc_curve(labels, outputs)
    maxFprIndex = np.where(fpr <= desiredFPR)[0][-1]
    fprBelow = fpr[maxFprIndex]
    fprAbove = fpr[maxFprIndex + 1]
    tprBelow = tpr[maxFprIndex]
    tprAbove = tpr[maxFprIndex + 1]
    tprAt = ((tprAbove - tprBelow) / (fprAbove - fprBelow) * (desiredFPR - fprBelow) + tprBelow)
    return tprAt, fpr, tpr

def predictTest(trainFeatures, trainLabels, testFeatures):
    imputer = KNNImputer(missing_values=-1, n_neighbors=3)
    trainFeatures_imputed = imputer.fit_transform(trainFeatures)
    testFeatures_imputed = imputer.transform(testFeatures)
    xgb_model = xgb.XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        learning_rate=0.1,
        max_depth=3,
        n_estimators=200,
        subsample=1.0
    )
    xgb_model.fit(trainFeatures_imputed, trainLabels)
    testOutputs = xgb_model.predict_proba(testFeatures_imputed)[:, 1]
    return testOutputs

if __name__ == "__main__":
    data = np.loadtxt('./spamTrain1.csv', delimiter=',')
    shuffleIndex = np.arange(np.shape(data)[0])
    np.random.shuffle(shuffleIndex)
    data = data[shuffleIndex, :]
    features = data[:, :-1]
    labels = data[:, -1]
    print("10-fold cross-validation mean AUC: ", np.mean(aucCV(features, labels)))
    trainFeatures = features[0::2, :]
    trainLabels = labels[0::2]
    testFeatures = features[1::2, :]
    testLabels = labels[1::2]
    testOutputs = predictTest(trainFeatures, trainLabels, testFeatures)
    print("Test set AUC: ", roc_auc_score(testLabels, testOutputs))
    tprAtDesiredFPR, fpr, tpr = tprAtFPR(testLabels, testOutputs, desiredFPR)
    print(f'TPR at FPR = .01: {tprAtDesiredFPR}')
    sortIndex = np.argsort(testLabels)
    nTestExamples = testLabels.size
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(nTestExamples), testLabels[sortIndex], 'b.')
    plt.xlabel('Sorted example number')
    plt.ylabel('Target')
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(nTestExamples), testOutputs[sortIndex], 'r.')
    plt.xlabel('Sorted example number')
    plt.ylabel('Output (predicted target)')
    plt.show()