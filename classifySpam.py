# -*- coding: utf-8 -*-
"""
Demo of 10-fold cross-validation using Random Forest on spam data

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

def aucCV(features, labels):
    # Define the pipeline with KNN imputation and Random Forest
    model = make_pipeline(KNNImputer(missing_values=-1, n_neighbors=5),
                          RandomForestClassifier(n_estimators=100, random_state=42))
    
    # Perform cross-validation with AUC as the scoring metric
    scores = cross_val_score(model, features, labels, cv=10, scoring='roc_auc')
    
    return scores

def predictTest(trainFeatures, trainLabels, testFeatures):
    # Define the pipeline with KNN imputation and Random Forest
    model = make_pipeline(KNNImputer(missing_values=-1, n_neighbors=5),
                          RandomForestClassifier(n_estimators=100, random_state=42))
    
    # Train the model
    model.fit(trainFeatures, trainLabels)
    
    # Return the predicted probabilities for class 1 (spam)
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
