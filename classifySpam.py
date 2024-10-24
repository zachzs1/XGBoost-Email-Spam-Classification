# -*- coding: utf-8 -*-
"""
Demo of XGBoost with Hyperparameter Tuning and multiple evaluation metrics using 5-fold cross-validation on spam data
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb
import optuna

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

    # Using StratifiedKFold for cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(xgb_model, features_imputed, labels, cv=skf, scoring='roc_auc', n_jobs=-1)
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

def plot_feature_importance(model, feature_names):
    # Get the feature importance values
    importance = model.get_booster().get_score(importance_type='gain')
    # Sort the features by importance
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.barh([x[0] for x in sorted_importance], [x[1] for x in sorted_importance])
    plt.xlabel("Feature Importance (Gain)")
    plt.ylabel("Feature")
    plt.title("Feature Importance using XGBoost")
    plt.gca().invert_yaxis()
    plt.show()

def tune_hyperparameters(features, labels):
    # Define the XGBoost model (the base estimator)
    xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)

    # Define the parameter grid to search over
    def objective(trial):
        # Define the hyperparameter search space
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'n_estimators': trial.suggest_int('n_estimators', 100, 700),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 1.5)
        }

        # Train the model with cross-validation and return the mean AUC
        model = xgb.XGBClassifier(eval_metric='logloss', random_state=42, **params)
        auc = cross_val_score(model, features, labels, cv=5, scoring='roc_auc', n_jobs=-1).mean()
        return auc

    # Create an Optuna study (without n_jobs)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())

    # Optimize the study with multiple jobs using `n_jobs`
    study.optimize(objective, n_trials=2000, n_jobs=-1)  # Uses all available CPU cores for trials

    print(f"Best parameters found: {study.best_params}")
    print(f"Best AUC score: {study.best_value:.4f}")

    # Return the best estimator with optimal parameters
    return xgb.XGBClassifier(eval_metric='logloss', random_state=42, **study.best_params)


def predictTest(trainFeatures, trainLabels, testFeatures):
    # Impute missing values using KNNImputer
    imputer = KNNImputer(missing_values=-1, n_neighbors=3)
    trainFeatures_imputed = imputer.fit_transform(trainFeatures)
    testFeatures_imputed = imputer.transform(testFeatures)
    
    # Tune hyperparameters using Optuna
    best_model = tune_hyperparameters(trainFeatures_imputed, trainLabels)
    
    # Fit the best model on the entire training set
    best_model.fit(trainFeatures_imputed, trainLabels)
    
    # Predict the probabilities of the positive class for the test set
    testOutputs = best_model.predict_proba(testFeatures_imputed)[:, 1]
    plot_feature_importance(best_model, feature_names=[f"Feature_{i}" for i in range(trainFeatures.shape[1])])
    return testOutputs

if __name__ == "__main__":
    train1DataFilename = 'spamTrain1.csv'
    train2DataFilename = 'spamTrain2.csv'
    train1Data = np.loadtxt(train1DataFilename, delimiter=',')
    train2Data = np.loadtxt(train2DataFilename, delimiter=',')
    data = np.r_[train1Data, train2Data]
    shuffleIndex = np.arange(np.shape(data)[0])
    np.random.shuffle(shuffleIndex)
    data = data[shuffleIndex, :]
    features = data[:, :-1] 
    labels = data[:, -1]
    print("5-fold Stratified Cross-Validation mean AUC: ", np.mean(aucCV(features, labels)))
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
