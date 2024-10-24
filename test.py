# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

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
trainFeatures = features[0::2, :]
trainLabels = labels[0::2]
testFeatures = features[1::2, :]
testLabels = labels[1::2]
testOutputs = predictTest(trainFeatures, trainLabels, testFeatures)
# Initializing imputers
knn_imputer = KNNImputer(n_neighbors=3)
mice_imputer = IterativeImputer(random_state=42)

# Lists to store results
knn_auc_scores = []
knn_tpr_scores = []
mice_auc_scores = []
mice_tpr_scores = []

desired_fpr = 0.01

# Function to calculate TPR at a given FPR
def tpr_at_fpr(labels, outputs, desired_fpr):
    fpr, tpr, _ = roc_curve(labels, outputs)
    max_fpr_index = np.where(fpr <= desired_fpr)[0][-1]
    fpr_below = fpr[max_fpr_index]
    fpr_above = fpr[max_fpr_index + 1]
    tpr_below = tpr[max_fpr_index]
    tpr_above = tpr[max_fpr_index + 1]
    tpr_at = ((tpr_above - tpr_below) / (fpr_above - fpr_below) * (desired_fpr - fpr_below) + tpr_below)
    return tpr_at

# Running KNN Imputer 5 times
for i in range(5):
    # KNN Imputation
    X_imputed_knn = knn_imputer.fit_transform(X)
    
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_imputed_knn, y, test_size=0.2, random_state=42)
    
    # Training the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Making predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculating AUC and TPR at desired FPR
    auc_score = roc_auc_score(y_test, y_pred_proba)
    tpr_score = tpr_at_fpr(y_test, y_pred_proba, desired_fpr)
    
    knn_auc_scores.append(auc_score)
    knn_tpr_scores.append(tpr_score)

# Running MICE Imputer 5 times
for i in range(5):
    # MICE Imputation
    X_imputed_mice = mice_imputer.fit_transform(X)
    
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_imputed_mice, y, test_size=0.2, random_state=42)
    
    # Training the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Making predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculating AUC and TPR at desired FPR
    auc_score = roc_auc_score(y_test, y_pred_proba)
    tpr_score = tpr_at_fpr(y_test, y_pred_proba, desired_fpr)
    
    mice_auc_scores.append(auc_score)
    mice_tpr_scores.append(tpr_score)

# Calculating mean and standard deviation for AUC and TPR for both imputers
knn_auc_mean = np.mean(knn_auc_scores)
knn_auc_std = np.std(knn_auc_scores)
knn_tpr_mean = np.mean(knn_tpr_scores)
knn_tpr_std = np.std(knn_tpr_scores)

mice_auc_mean = np.mean(mice_auc_scores)
mice_auc_std = np.std(mice_auc_scores)
mice_tpr_mean = np.mean(mice_tpr_scores)
mice_tpr_std = np.std(mice_tpr_scores)

# Printing results
print(f'KNN Imputer - AUC Mean: {knn_auc_mean:.2f}, AUC Std Dev: {knn_auc_std:.2f}')
print(f'KNN Imputer - TPR Mean: {knn_tpr_mean:.2f}, TPR Std Dev: {knn_tpr_std:.2f}')
print(f'MICE Imputer - AUC Mean: {mice_auc_mean:.2f}, AUC Std Dev: {mice_auc_std:.2f}')
print(f'MICE Imputer - TPR Mean: {mice_tpr_mean:.2f}, TPR Std Dev: {mice_tpr_std:.2f}')
