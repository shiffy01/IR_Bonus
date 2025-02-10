import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Step 1: Load the dataset
df = pd.read_excel('BERT_file.xlsx')

# Step 2: Convert BERT vectors from string to list of floats
X = np.array([eval(v) for v in df['sbert_embedded']])  # Assuming 'bert_vector' is a string representation of the list
y = df['label encoded'].values  # Assuming 'label' is the column with actual labels

# Step 3: Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    X, y, df.index, test_size=0.2, random_state=42)  # We also return the indices of the train and test sets

# Step 4: Scale the features to improve model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Set up GridSearchCV to tune hyperparameters 'C' and 'gamma' for the RBF kernel
param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 0.5, 1, 'scale']}
grid_search = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid, cv=10, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Step 6: Train the model on the training set using the best parameters
best_model.fit(X_train_scaled, y_train)

# Step 7: Make predictions on both the training and test sets
train_predictions = best_model.predict(X_train_scaled)
test_predictions = best_model.predict(X_test_scaled)

# Get probabilities for confidence scores (for both training and test sets)
train_confidence_scores = best_model.predict_proba(X_train_scaled)
test_confidence_scores = best_model.predict_proba(X_test_scaled)

# Step 8: Evaluate the model on the test set
accuracy = accuracy_score(y_test, test_predictions)
precision = precision_score(y_test, test_predictions, average='weighted')
recall = recall_score(y_test, test_predictions, average='weighted')
f1 = f1_score(y_test, test_predictions, average='weighted')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, test_predictions)
print('Confusion Matrix:')
print(conf_matrix)

# Step 9: Add predictions and confidence scores to the original DataFrame for both train and test sets
df.loc[train_indices, 'predictions'] = train_predictions
df.loc[test_indices, 'predictions'] = test_predictions

df.loc[train_indices, 'confidence_scores'] = [max(score) for score in train_confidence_scores]
df.loc[test_indices, 'confidence_scores'] = [max(score) for score in test_confidence_scores]

# Step 10: Save the updated DataFrame with predictions and confidence scores to a new Excel file
df.to_excel('svm_cross_val_sbert1.xlsx', index=False)

# Step 11: Visualize the confidence score distribution for training data
train_conf_scores = best_model.predict_proba(X_train_scaled)
plt.hist([max(conf) for conf in train_conf_scores], bins=50)
plt.title('Training Data Confidence Scores Distribution')
plt.xlabel('Confidence')
plt.ylabel('Frequency')
plt.show()
