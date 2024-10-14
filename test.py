import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('./spamTrain1.csv')

# Replace -1 (missing values) with NaN, to handle them separately if needed
data.replace(-1, np.nan, inplace=True)

# Separate features (columns) and target (spam or not spam)
features = data.iloc[:, :-1]  # All but last column are features
target = data.iloc[:, -1]     # Last column is the target (spam or not)

# Create masks for spam and not spam
spam_mask = target == 1
not_spam_mask = target == 0

# Initialize dictionaries to store spam and not spam percentages for each feature
spam_percentages = {}
not_spam_percentages = {}

# Loop through each feature (column)
for feature in features.columns:
    # Calculate the number of non-zero appearances of the feature in spam and not spam
    spam_appearances = features.loc[spam_mask, feature].fillna(0).gt(0).sum()  # Non-zero entries in spam
    not_spam_appearances = features.loc[not_spam_mask, feature].fillna(0).gt(0).sum()  # Non-zero entries in not spam
    
    # Total spam and not spam emails
    total_spam = spam_mask.sum()
    total_not_spam = not_spam_mask.sum()
    
    # Calculate percentages for spam and not spam
    spam_percentage = (spam_appearances / total_spam) * 100
    not_spam_percentage = (not_spam_appearances / total_not_spam) * 100
    
    # Store the results
    spam_percentages[feature] = spam_percentage
    not_spam_percentages[feature] = not_spam_percentage

# Convert the results to a DataFrame for easier viewing
results = pd.DataFrame({
    'Feature': features.columns,
    'Spam Percentage': [spam_percentages[feature] for feature in features.columns],
    'Not Spam Percentage': [not_spam_percentages[feature] for feature in features.columns]
})

# Display the results
print(results)
