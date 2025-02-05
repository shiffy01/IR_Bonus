import pandas as pd

# Load the Excel file
file_path = "sentences_shiffy.xlsx"  # Replace with your actual file path
xls = pd.ExcelFile(file_path)

# Dictionary to store label counts
label_counts = {i: 0 for i in range(5)}
total_lines = 0

# Process each sheet
for sheet_name in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name)

    # Ensure column exists
    if "Label By Hand" in df.columns:
        filtered_df = df[df["Label By Hand"].notna()]  # Filter non-empty labels
        total_lines += len(filtered_df)

        # Count occurrences of each label
        for i in range(5):
            label_counts[i] += (filtered_df["Label By Hand"] == i).sum()

# Print results
print(f"Total lines taken: {total_lines}")
for label, count in label_counts.items():
    print(f"Label {label}: {count}")
