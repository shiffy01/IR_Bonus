import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import ast

# Read Excel file
df = pd.read_excel('BERT_file.xlsx')

# Function to convert string BERT vectors to numpy arrays
def string_to_vector(bert_string):
    return np.array(ast.literal_eval(bert_string))

# Apply the conversion to the 'bert_vector' column
df['bert_vector'] = df['bert_vector'].apply(string_to_vector)

# Keep track of the original indices
df['index'] = df.index

# Split features and labels
X = np.stack(df['bert_vector'].values)
y = df['label encoded'].values

# Encode the labels into one-hot format
y = to_categorical(y, num_classes=5)

# Split the data into training (80%), validation (10%), and test (20%)
X_train, X_temp, y_train, y_temp, train_indices, temp_indices = train_test_split(
    X, y, df['index'], test_size=0.2, random_state=42)

X_val, X_test, y_val, y_test, val_indices, test_indices = train_test_split(
    X_temp, y_temp, temp_indices, test_size=0.25, random_state=42)  # 20% test, 10% validation

# Define the ANN model
model = Sequential([
    Dense(64, input_dim=X.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(5, activation='softmax')  # 5 output classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val),
                    callbacks=[early_stopping])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

# Predict on validation set
y_val_pred = model.predict(X_val)
val_pred_classes = np.argmax(y_val_pred, axis=1)

# Confidence scores for validation
val_confidence = np.max(y_val_pred, axis=1)

# Save predictions, confidence scores, and original data to Excel
df_val = pd.DataFrame({
    'Sentence': df.loc[val_indices, 'Sentence'],
    'True Label': np.argmax(y_val, axis=1),
    'Predicted Label': val_pred_classes,
    'Confidence': val_confidence
})

df_val.to_excel('ann_bert.xlsx', index=False)
