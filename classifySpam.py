# -*- coding: utf-8 -*-
"""
Demo of 10-fold cross-validation using Random Forest on spam data

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb

desiredFPR = 0.01

def aucCV(features, labels):
    # Define the pipeline with KNN imputation and XGBoost using best parameters
    model = make_pipeline(
        KNNImputer(missing_values=-1, n_neighbors=3),  # Using n_neighbors=3 as found from grid search
        xgb.XGBClassifier(
            eval_metric='logloss', 
            random_state=42,
            learning_rate=0.1,  # Using the best learning rate
            max_depth=5,        # Using the best max depth
            n_estimators=100,    # Using the best number of estimators
            subsample=1.0        # Using the best subsample
        )
    )
    
    # Perform 10-fold cross-validation and return mean AUC score
    auc_scores = cross_val_score(model, features, labels, cv=10, scoring='roc_auc', n_jobs=-1)
    return np.mean(auc_scores)

def tprAtFPR(labels, outputs, desiredFPR):
    fpr, tpr, thres = roc_curve(labels, outputs)
    # True positive rate for highest false positive rate <= desiredFPR
    maxFprIndex = np.where(fpr <= desiredFPR)[0][-1]
    fprBelow = fpr[maxFprIndex]
    fprAbove = fpr[maxFprIndex + 1]
    # Find TPR at exactly desired FPR by linear interpolation
    tprBelow = tpr[maxFprIndex]
    tprAbove = tpr[maxFprIndex + 1]
    tprAt = ((tprAbove - tprBelow) / (fprAbove - fprBelow) * (desiredFPR - fprBelow)
             + tprBelow)
    return tprAt, fpr, tpr

def predictTest(trainFeatures, trainLabels, testFeatures):
    # Define the pipeline with KNN imputation and XGBoost using best parameters
    model = make_pipeline(
        KNNImputer(missing_values=-1, n_neighbors=3),  # Using n_neighbors=3 as found from grid search
        xgb.XGBClassifier(
            eval_metric='logloss',
            random_state=42,
            learning_rate=0.1,  # Using the best learning rate
            max_depth=5,        # Using the best max depth
            n_estimators=100,    # Using the best number of estimators
            subsample=1.0        # Using the best subsample
        )
    )
    
    # Train the model with the training data
    model.fit(trainFeatures, trainLabels)
    
    # Predict probabilities for the test set
    testOutputs = model.predict_proba(testFeatures)[:, 1]
    
    return testOutputs


# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    data = np.loadtxt('./spamTrain1.csv', delimiter=',')
    
    # Randomly shuffle rows of dataset, then separate labels (last column)
    shuffleIndex = np.arange(np.shape(data)[0])
    np.random.shuffle(shuffleIndex)
    data = data[shuffleIndex, :]
    features = data[:, :-1]
    labels = data[:, -1]
    
    # Evaluate classifier accuracy using 10-fold cross-validation
    print("10-fold cross-validation mean AUC: ",
          np.mean(aucCV(features, labels)))
    
    # Arbitrarily choose all odd samples as train set and all even as test set
    trainFeatures = features[0::2, :]
    trainLabels = labels[0::2]
    testFeatures = features[1::2, :]
    testLabels = labels[1::2]
    testOutputs = predictTest(trainFeatures, trainLabels, testFeatures)
    print("Test set AUC: ", roc_auc_score(testLabels, testOutputs))
    tprAtDesiredFPR, fpr, tpr = tprAtFPR(testLabels, testOutputs, desiredFPR)
    print(f'TPR at FPR = .01 {tprAtDesiredFPR}')
    
    # Examine outputs compared to labels
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
