# import pandas as pd
# import numpy as np
# from ast import literal_eval
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report
#
# # Load the Excel file
# file_path = "with_prediction_good.xlsx"  # Change this to your file path
# df = pd.read_excel(file_path)
#
# # Convert the "sbert embedding" column from string to actual list
# df["sbert embedding string"] = df["sbert_embedded"].apply(literal_eval)
#
# # Convert the list embeddings into a NumPy array
# X = np.array(df["sbert embedding string"].tolist())
#
# # Extract encoded labels
# y = df["label encoded"].values  # Change "encoded label" to the actual column name
#
# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Train Logistic Regression model
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)
#
# # Predict on test data
# y_pred = model.predict(X_test)
#
# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.4f}")
# print("Classification Report:\n", classification_report(y_test, y_pred))

#
# import pandas as pd
# import numpy as np
# from ast import literal_eval
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report
#
# # Load the Excel file
# file_path = "with_prediction_good.xlsx"  # Change this to your file path
# df = pd.read_excel(file_path)
#
# # Convert the "sbert embedding" column from string to actual list
# df["sbert embedding string"] = df["sbert_embedded"].apply(literal_eval)
#
# # Convert the list embeddings into a NumPy array
# X = np.array(df["sbert embedding string"].tolist())
#
# # Extract encoded labels
# y = df["label encoded"].values  # Change "label encoded" to the actual column name
#
# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Train Logistic Regression model
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)
#
# # Predict on test data
# y_pred = model.predict(X_test)
# y_pred_prob = model.predict_proba(X_test)  # Get probability scores
#
# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.4f}")
# print("Classification Report:\n", classification_report(y_test, y_pred))
#
# # Add predictions and probabilities back to the DataFrame
# df.loc[df.index[y_test.index], "logreg_prediction"] = y_pred
# df.loc[df.index[y_test.index], "logreg_prediction_score"] = y_pred_prob.max(axis=1)  # Highest probability for each prediction
#
# # Save the updated DataFrame back to Excel
# output_file = "with_predictions_updated.xlsx"
# df.to_excel(output_file, index=False)
#
# print(f"Predictions saved to {output_file}")


import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the Excel file
file_path = "with_prediction_good.xlsx"  # Change to your actual file path
df = pd.read_excel(file_path)

# Convert the "sbert embedding" column from string to actual list
df["sbert embedding string"] = df["sbert_embedded"].apply(literal_eval)

# Convert embeddings to NumPy array
X = np.array(df["sbert embedding string"].tolist())

# Extract labels
y = df["label encoded"].values

# Define the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Set up 10-Fold Cross-Validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Perform 10-Fold Cross-Validation to get predictions
y_pred = cross_val_predict(model, X, y, cv=cv, method="predict")
y_pred_prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")

# Evaluate the model
accuracy = accuracy_score(y, y_pred)
print(f"Cross-Validation Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y, y_pred))

# Add predictions and probabilities back to the DataFrame
df["logreg_prediction"] = y_pred
df["logreg_prediction_score"] = y_pred_prob.max(axis=1)  # Highest probability for each prediction

# Save the updated DataFrame to the same file
df.to_excel(file_path, index=False, engine="openpyxl")

print(f"Predictions saved to {file_path}")
