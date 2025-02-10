#encodes the sentences (with words from 1 category)
import pandas as pd
from sentence_transformers import SentenceTransformer #Python library that provides an easy-to-use API to work with SBERT models.

# Load the SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load the Excel file
file_path = "with_prediction_good.xlsx"
df = pd.read_excel(file_path)

# Check if the "Sentence" column exists
if "Sentence" not in df.columns:
    raise ValueError("The column 'Sentence' is not found in the dataset!")

# Generate SBERT embeddings
df["sbert_embedded"] = df["Sentence"].apply(lambda x: model.encode(str(x)).tolist())

# Save the updated file (overwrite the existing one)
df.to_excel(file_path, index=False)

print(f"SBERT embeddings added and saved in '{file_path}'.")
