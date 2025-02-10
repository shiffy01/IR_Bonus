#from the big file with all the sentences extracts the sentences that we labled by hand -part A
import pandas as pd

# Load the Excel file
file_path = "sentences_shiffy.xlsx"
xls = pd.ExcelFile(file_path)

# Dictionary to store label counts
label_counts = {i: 0 for i in range(5)}
total_lines = 0
filtered_data = []  # List to store filtered rows

# Process each sheet
for sheet_name in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name)

    # Ensure column exists
    if "Label By Hand" in df.columns:
        filtered_df = df[df["Label By Hand"].notna()]  # Filter non-empty labels
        total_lines += len(filtered_df)
        filtered_data.append(filtered_df)  # Store filtered data

        # Count occurrences of each label
        for i in range(5):
            label_counts[i] += (filtered_df["Label By Hand"] == i).sum()

# Combine all filtered data into a single DataFrame
if filtered_data:
    final_df = pd.concat(filtered_data, ignore_index=True)
    # Save to a new Excel file
    output_file = "filtered_data_partA_both.xlsx"
    final_df.to_excel(output_file, index=False)
    print(f"Filtered data saved to {output_file}")

# Print results
print(f"Total lines taken: {total_lines}")
for label, count in label_counts.items():
    print(f"Label {label}: {count}")
