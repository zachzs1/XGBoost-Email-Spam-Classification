import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb
import optuna
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV
import warnings
warnings.filterwarnings("ignore")

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
    study.optimize(objective, n_trials=500, n_jobs=-1)  # Uses all available CPU cores for trials

    print(f"Best parameters found: {study.best_params}")
    print(f"Best AUC score: {study.best_value:.4f}")

    # Return the best estimator with optimal parameters
    return xgb.XGBClassifier(eval_metric='logloss', random_state=42, **study.best_params)

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.feature_selection import RFECV
import xgboost as xgb
import numpy as np

# Create a pipeline with imputation, feature selection, and the classifier
def predictTest(trainFeatures, trainLabels, testFeatures):
    # Create individual models
    initial_xgb_model = xgb.XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        learning_rate=0.1,
        max_depth=3,
        n_estimators=200,
        subsample=1.0,
        n_jobs=-1
    )
    
    # Use RFECV for robust feature selection inside a pipeline
    feature_selector = RFECV(estimator=initial_xgb_model, step=1, cv=StratifiedKFold(5), scoring='roc_auc', n_jobs=-1)
    
    # Create a final model
    best_model = xgb.XGBClassifier(
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

    # Create a pipeline with KNNImputer, RFECV for feature selection, and the classifier
    pipeline = Pipeline([
        ('imputer', KNNImputer(missing_values=-1, n_neighbors=7)),
        ('feature_selection', feature_selector),
        ('classifier', best_model)
    ])

    # Cross-validation setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cross_val_predictions = np.zeros(trainLabels.shape)

    # Perform cross-validation manually
    for train_index, val_index in skf.split(trainFeatures, trainLabels):
        X_train, X_val = trainFeatures[train_index], trainFeatures[val_index]
        y_train, y_val = trainLabels[train_index], trainLabels[val_index]

        # Fit the pipeline on the training fold
        pipeline.fit(X_train, y_train)

        # Predict probabilities for the validation fold
        cross_val_predictions[val_index] = pipeline.predict_proba(X_val)[:, 1]

    # Fit the final pipeline on the entire training set
    pipeline.fit(trainFeatures, trainLabels)

    # Predict the probabilities of the positive class for the test set
    testOutputs = pipeline.predict_proba(testFeatures)[:, 1]
    return testOutputs

def visualize_selected_features(total_features, important_features):
    """
    Visualize all features and highlight the selected important features.
    
    Parameters:
    - total_features: The total number of features in the dataset.
    - important_features: A list of indices of the selected important features.
    """
    plt.figure(figsize=(12, 6))
    all_features = np.arange(total_features)
    
    # Plot all features
    plt.scatter(all_features, np.zeros_like(all_features), color='gray', label='All Features')
    
    # Highlight important features
    plt.scatter(important_features, np.zeros_like(important_features), color='red', label='Important Features', s=100)
    
    # Plot settings
    plt.title("Visualization of Selected Important Features")
    plt.xlabel("Feature Index")
    plt.yticks([])  # Remove y-axis labels
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    train1DataFilename = 'spamTrain1.csv'
    train2DataFilename = 'spamTrain2.csv'
    train1Data = np.loadtxt(train1DataFilename, delimiter=',')
    train2Data = np.loadtxt(train2DataFilename, delimiter=',')
    data = np.r_[train1Data, train2Data]
    for i in range(20):
        shuffleIndex = np.arange(np.shape(data)[0])
        np.random.shuffle(shuffleIndex)
        data = data[shuffleIndex, :]
        features = data[:, :-1] 
        labels = data[:, -1]
        #print("5-fold Stratified Cross-Validation mean AUC: ", np.mean(aucCV(features, labels)))
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
