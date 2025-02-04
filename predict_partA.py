import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Path to model
model_path = r"C:\Users\rache\Desktop\איחזור מידע\hw4\fine_tuned_bert\content\fine_tuned_bert"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=5)
model.to(device)
model.eval()

# Load Excel file
file_path = "filtered_data_partA.xlsx"  # Replace with your actual file path
df = pd.read_excel(file_path)

# Ensure required columns exist
if "Sentence" not in df.columns or "Label By Hand" not in df.columns:
    raise ValueError("Excel file must contain 'Sentence' and 'Label By Hand' columns.")

# Filter sentences with labels
df = df[df["Label By Hand"].notna()]
texts = df["Sentence"].astype(str).tolist()
true_labels = df["Label By Hand"].astype(int).tolist()

# Tokenization function
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.argmax(outputs.logits, dim=1).cpu().item()

# Get predictions
predictions = [predict(text) for text in texts]

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
