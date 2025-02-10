import pandas as pd

# Load both files
file1_path = "BERT_file.xlsx"  # Replace with your actual file
file2_path = "ann_sbert.xlsx"  # Replace with your actual file

df1 = pd.read_excel(file1_path)
df2 = pd.read_excel(file2_path)

# Merge on "Sentence" column
merged_df = df1.merge(df2, on="Sentence", how="left")  # 'left' keeps all rows from file1

# Save the merged file
merged_df.to_excel("merged_fileSBERT.xlsx", index=False, engine="openpyxl")

print("Files merged and saved as merged_filesBert.xlsx")
