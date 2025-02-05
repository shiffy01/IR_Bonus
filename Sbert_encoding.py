# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer #Python library that provides an easy-to-use API to work with SBERT models.
# from nltk.tokenize import sent_tokenize, word_tokenize
#
# # Load SBERT model
# model = SentenceTransformer("all-MiniLM-L6-v2")  # Change model if needed
#
# # Load Excel file
# file_path = "with_prediction_good.xlsx"
# xls = pd.ExcelFile(file_path)
#
# # Function to split long text into smaller chunks
# def split_into_chunks(text, max_words=100):
#     """Splits text into smaller overlapping chunks (sentence-based)."""
#     sentences = sent_tokenize(text)  # Split by sentence
#     chunks, current_chunk = [], []
#
#     for sentence in sentences:
#         words = word_tokenize(sentence)
#         if len(current_chunk) + len(words) > max_words:
#             chunks.append(" ".join(current_chunk))
#             current_chunk = words
#         else:
#             current_chunk.extend(words)
#
#     if current_chunk:
#         chunks.append(" ".join(current_chunk))
#
#     return chunks
#
# # Function to embed long text
# def embed_long_text(text):
#     """Embeds long text by splitting into chunks and averaging embeddings."""
#     chunks = split_into_chunks(str(text))  # Convert text to string & split
#     embeddings = np.array([model.encode(chunk) for chunk in chunks])
#     return embeddings.mean(axis=0).tolist()  # Average the embeddings
#
# # Process each sheet and save back to the same file
# with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
#     for sheet_name in xls.sheet_names:
#         df = xls.parse(sheet_name)  # Read sheet
#         if "Sentence" in df.columns:  # Ensure column exists
#             df["sbert_embedded"] = df["Sentence"].apply(embed_long_text)  # Embed sentences
#         df.to_excel(writer, sheet_name=sheet_name, index=False)  # Save back to the same file
#
# print(f"Embedding completed. Results saved to {file_path}")


import pandas as pd
from sentence_transformers import SentenceTransformer #Python library that provides an easy-to-use API to work with SBERT models.

# Load the SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")  # You can change this model if needed

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
