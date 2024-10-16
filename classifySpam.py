# -*- coding: utf-8 -*-
"""
Demo of 10-fold cross-validation using Random Forest on spam data

@author: Kevin S. Xu
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb

desiredFPR = 0.01

def aucCV(features, labels):
    # Define the pipeline with KNN imputation and XGBoost with GPU support
    model = make_pipeline(
        KNNImputer(missing_values=-1),  # KNN Imputer to handle missing values
        xgb.XGBClassifier(tree_method='gpu_hist', eval_metric='logloss', random_state=42)  # GPU acceleration
    )
    
    # Define the parameter grid for XGBoost and KNNImputer
    param_grid = {
        'knnimputer__n_neighbors': [3, 5, 7],  # KNN neighbors
        'xgbclassifier__n_estimators': [100, 200, 300],  # Number of boosting rounds
        'xgbclassifier__max_depth': [3, 5, 7],  # Max depth of trees
        'xgbclassifier__learning_rate': [0.01, 0.1, 0.2],  # Learning rate for gradient boosting
        'xgbclassifier__subsample': [0.8, 1.0]  # Fraction of samples used for training each tree
    }
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='roc_auc',  # Use AUC score as the evaluation metric
        n_jobs=-1,  # Use all available cores
        verbose=0  # Suppress detailed output
    )
    
    # Perform grid search cross-validation
    grid_search.fit(features, labels)
    
    # Return the best score from the grid search
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best AUC score: {grid_search.best_score_}")
    
    return grid_search.best_score_

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
    # Define the pipeline with KNN imputation and XGBoost with GPU support
    model = make_pipeline(
        KNNImputer(missing_values=-1),  # KNN Imputer
        xgb.XGBClassifier(tree_method='gpu_hist', eval_metric='logloss', random_state=42)  # GPU acceleration
    )
    
    # Define the parameter grid for XGBoost and KNNImputer
    param_grid = {
        'knnimputer__n_neighbors': [3, 5, 7],
        'xgbclassifier__n_estimators': [100, 200, 300],
        'xgbclassifier__max_depth': [3, 5, 7],
        'xgbclassifier__learning_rate': [0.01, 0.1, 0.2],
        'xgbclassifier__subsample': [0.8, 1.0]
    }
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    # Train the model with grid search
    grid_search.fit(trainFeatures, trainLabels)
    
    # Predict probabilities for the test set using the best model
    testOutputs = grid_search.best_estimator_.predict_proba(testFeatures)[:, 1]
    
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
    tprAtDesiredFPR,fpr,tpr = tprAtFPR(testLabels,testOutputs,desiredFPR)
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
