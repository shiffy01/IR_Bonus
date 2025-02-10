import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import ast

# Read Excel file
df = pd.read_excel('BERT_file.xlsx')


# Function to convert string BERT vectors to numpy arrays
def string_to_vector(bert_string):
    return np.array(ast.literal_eval(bert_string))


# Convert 'bert_vector' column to numpy arrays
df['sbert_embedded'] = df['sbert_embedded'].apply(string_to_vector)

# Keep track of the original indices
df['index'] = df.index

# Split features and labels
X = np.stack(df['sbert_embedded'].values)
y = df['label encoded'].values

# Encode labels into one-hot format
y = to_categorical(y, num_classes=5)

# Split data into training (80%), validation (10%), and test (10%)
X_train, X_temp, y_train, y_temp, train_indices, temp_indices = train_test_split(
    X, y, df['index'], test_size=0.2, random_state=42)

X_val, X_test, y_val, y_test, val_indices, test_indices = train_test_split(
    X_temp, y_temp, temp_indices, test_size=0.1, random_state=42)  # 10% test, 10% validation

# Define the ANN model
model = Sequential([
    Dense(32, input_dim=X.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(5, activation='softmax')  # 5 output classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Filepath to save the best model
checkpoint_filepath = 'best_model_sbert.h5'

# Callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', patience=5, restore_best_weights=True)

model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy', mode='max',
                                   save_best_only=True, verbose=1)

# Train the model with callbacks
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val),
                    callbacks=[early_stopping, model_checkpoint])

# Load the best model (ensures best accuracy model is used)
model.load_weights(checkpoint_filepath)

# Find the epoch with the highest validation accuracy
best_epoch = np.argmax(history.history['val_accuracy']) + 1  # +1 because epochs start from 1

print(f"Best Epoch Used: {best_epoch}")

# Predict on validation set
y_val_pred = model.predict(X_val)
val_pred_classes = np.argmax(y_val_pred, axis=1)
val_confidence = np.max(y_val_pred, axis=1)

# Save predictions, confidence scores, and best epoch to Excel
df_val = pd.DataFrame({
    'Sentence': df.loc[val_indices, 'Sentence'],
    'True Label': np.argmax(y_val, axis=1),
    'Predicted Label': val_pred_classes,
    'Confidence': val_confidence,
})

df_val.to_excel('ann_bert.xlsx', index=False)

# Evaluate the best model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")
