import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Load the Excel file
file_path = "BERT_file.xlsx"  # Replace with your actual file
df = pd.read_excel(file_path)

# Identify prediction and score columns
prediction_cols = [col for col in df.columns if "prediction" in col.lower()]
score_cols = [col for col in df.columns if "score" in col.lower()]
print(prediction_cols)
print(score_cols)

# Function to calculate majority label with tie-breaking based on score
def majority_label_with_score(row):
    counts = Counter(row[prediction_cols])  # Count occurrences of each label
    labels = [label for label, count in counts.items() if count == max(counts.values())]  # Identify tied labels
    print(counts)
    print(labels)
    if len(labels) == 1:
        return labels[0]  # No tie, return the majority label
    else:
        # Tie-breaking: calculate the average score for each tied label
        avg_scores = {label: 0 for label in labels} #sets the counters with 0

        # Loop through the prediction columns and match them to the score columns
        for i, label in enumerate(row[prediction_cols]):
            if label in labels:
                score_col_idx = prediction_cols.index(prediction_cols[i])  # Find the corresponding score column index
                avg_scores[label] += row[score_cols[score_col_idx]]  # Add the score for this label

        # Return the label with the highest average score
        return max(avg_scores, key=avg_scores.get)


# Apply function to each row
df["Majority_Label"] = df.apply(majority_label_with_score, axis=1)

# Compare with true label
df["Correct_Prediction"] = (df["Majority_Label"] == df["label encoded"]).astype(int)

# Calculate accuracy
accuracy = df["Correct_Prediction"].mean()
print(f"Accuracy: {accuracy:.4f}")

# Save the updated DataFrame
df.to_excel("updated_predictions_with_score.xlsx", index=False, engine="openpyxl")

print("File saved as updated_predictions_with_score.xlsx")




# Plot 1: Accuracy per label
label_accuracy = df.groupby("label encoded")["Correct_Prediction"].mean()

plt.figure(figsize=(8, 6))
sns.barplot(x=label_accuracy.index, y=label_accuracy.values, palette="Blues")
plt.xlabel("Label")
plt.ylabel("Accuracy")
plt.title("Accuracy of Each Label")
plt.xticks(rotation=45)
plt.show()

# Plot 2: Distribution of Majority Labels
plt.figure(figsize=(8, 6))
sns.histplot(df["Majority_Label"], bins=5, kde=False, color="coral")
plt.xlabel("Majority Label")
plt.ylabel("Count")
plt.title("Distribution of Majority Labels")
plt.xticks(rotation=45)
plt.show()



#confusion matrix
labels = sorted(df["label encoded"].unique())
cm = confusion_matrix(df["label encoded"], df["Majority_Label"], labels=labels)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

#average score
avg_scores_per_label = df.groupby("label encoded")[score_cols].mean()

plt.figure(figsize=(10, 6))
sns.boxplot(data=avg_scores_per_label)
plt.xlabel("Label")
plt.ylabel("Score")
plt.title("Distribution of Scores for Each Label")
plt.show()

#corecct count
plt.figure(figsize=(8, 6))
sns.histplot(df["Correct_Prediction"], bins=2, kde=False, discrete=True, color="green")
plt.xticks([0, 1], ["Incorrect", "Correct"])
plt.xlabel("Prediction Outcome")
plt.ylabel("Count")
plt.title("Correct vs. Incorrect Predictions")
plt.show()

#score distrobution
plt.figure(figsize=(10, 6))
for score_col in score_cols:
    sns.kdeplot(df[score_col], label=score_col, fill=True, alpha=0.3)

plt.xlabel("Prediction Score")
plt.ylabel("Density")
plt.title("Score Distribution for All Models")
plt.legend()
plt.show()
