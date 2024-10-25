import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV

def predictTest(trainFeatures, trainLabels, testFeatures):
    initial_xgb_model = xgb.XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        learning_rate=0.059570899098437186,
        max_depth=3,
        n_estimators=190,
        subsample=0.6162218302874191,
        colsample_bytree=0.5238538843208448,
        gamma=0.14897009050327084,
        min_child_weight=1,
        reg_alpha=0.0950768790629103,
        reg_lambda=0.8399273699950727,
        n_jobs=-1
    )
    
    feature_selector = RFECV(estimator=initial_xgb_model, step=1, cv=StratifiedKFold(3), scoring='roc_auc', n_jobs=-1)
    pipeline = Pipeline([
        ('imputer', KNNImputer(missing_values=-1, n_neighbors=7)),
        ('feature_selection', feature_selector),
        ('classifier', initial_xgb_model)
    ])

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cross_val_predictions = np.zeros(trainLabels.shape)

    for train_index, val_index in skf.split(trainFeatures, trainLabels):
        X_train, X_val = trainFeatures[train_index], trainFeatures[val_index]
        y_train, y_val = trainLabels[train_index], trainLabels[val_index]
        pipeline.fit(X_train, y_train)
        cross_val_predictions[val_index] = pipeline.predict_proba(X_val)[:, 1]

    pipeline.fit(trainFeatures, trainLabels)
    testOutputs = pipeline.predict_proba(testFeatures)[:, 1]
    return testOutputs

desiredFPR = 0.01

def tprAtFPR(labels, outputs, desiredFPR):
    fpr, tpr, thres = roc_curve(labels, outputs)
    maxFprIndex = np.where(fpr <= desiredFPR)[0][-1]
    fprBelow = fpr[maxFprIndex]
    fprAbove = fpr[maxFprIndex + 1]
    tprBelow = tpr[maxFprIndex]
    tprAbove = tpr[maxFprIndex + 1]
    tprAt = ((tprAbove - tprBelow) / (fprAbove - fprBelow) * (desiredFPR - fprBelow) + tprBelow)
    return tprAt, fpr, tpr

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
