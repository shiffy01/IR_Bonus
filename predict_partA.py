#sends each sentence to the model from hw4 to see its results
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Path to model from HW4
model_path = r"C:\Users\rache\Desktop\איחזור מידע\hw4\fine_tuned_bert\content\fine_tuned_bert"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=5)
model.to(device)
model.eval()

# Load Excel file
file_path = "filtered_data_partA_both.xlsx"  # Replace with your actual file path
df = pd.read_excel(file_path)

# Ensure required columns exist
if "Sentence" not in df.columns or "Label By Hand" not in df.columns:
    raise ValueError("Excel file must contain 'Sentence' and 'Label By Hand' columns.")

# Filter sentences with labels
df = df[df["Label By Hand"].notna()]

# Prepare dataset by selecting 35 random sentences per label
df_sampled = pd.DataFrame()

# For each label, sample 35 sentences
for label in range(5):
    label_df = df[df["Label By Hand"] == label]
    sampled_df = label_df.sample(n=35, random_state=42)  # You can change the random_state for variability
    df_sampled = pd.concat([df_sampled, sampled_df])

# Shuffle the sampled dataframe
df_sampled = df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)

# Ensure that we are using only the sampled dataset
texts = df_sampled["Sentence"].astype(str).tolist()
true_labels = df_sampled["Label By Hand"].astype(int).tolist()

# Tokenization function
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    # Get the predicted label and the score (probability of the predicted label)
    predicted_label = torch.argmax(logits, dim=1).cpu().item()
    score = torch.softmax(logits, dim=1).max().cpu().item()  # Get the max probability
    return predicted_label, score

# Get predictions and scores
predictions = []
scores = []
for text in texts:
    pred, score = predict(text)
    predictions.append(pred)
    scores.append(score)

# Add predictions and scores as new columns in the dataframe
df_sampled['Prediction'] = predictions
df_sampled['Score'] = scores

# Compute metrics
accuracy = accuracy_score(true_labels, predictions)
report = classification_report(true_labels, predictions, target_names=[str(i) for i in range(5)])
conf_matrix = confusion_matrix(true_labels, predictions)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", conf_matrix)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(5), yticklabels=range(5))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Save the sampled dataset with predictions and scores to a new Excel file
df_sampled.to_excel("filtered_data_sampled_with_predictions_and_scores.xlsx", index=False)
