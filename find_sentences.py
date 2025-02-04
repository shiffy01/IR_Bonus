
import pandas as pd
import re
import nltk

nltk.download('vader_lexicon')


israel_words = [
    "Cabinet", "Colonizers", "Government", "Homeland", "Humanitarian Aid",
    "IDF", "Iron Dome", "Israel", "Israeli", "Jerusalem", "Jewish",
    "Knesset", "Mossad", "Netanyahu", "Occupation", "Occupied Territories",
    "Occupiers", "Parliament", "Settlers", "Tel Aviv", "Tel-Aviv",
    "West Bank", "West-Bank", "Zionism", "Zionist entity", "Zionist regime",
    "Zionist State", "cabinet", "colonizers", "government", "homeland",
    "humanitarian", "aid", "idf", "iron dome", "israel", "israeli",
    "jerusalem", "jewish", "knesset", "mossad", "netanyahu", "occupation",
    "occupied", "territories", "occupiers", "parliament", "settlers",
    "tel aviv", "tel-aviv", "west bank", "west-bank", "zionism",
    "zionist entity", "zionist", "regime", "zionist state"
]
palestine_words = [
    "Abbas", "Displaced", "Freedom fighters", "Gaza", "Gazans", "Hamas",
    "Hassan Nasrallah", "Hezbollah", "Houthis", "Humanitarian Crisis", "Intifada",
    "Iran", "Muhammad Sinuar", "Naim Qassem", "Nakba", "Nukhba", "Oppressed",
    "Organization", "Palestine", "Palestinians", "PLO", "Refugees", "Resistance",
    "Resisters", "Sinuar", "Terrorists", "Tyrants", "Victims", "abbas", "displaced",
    "freedom", "fighters", "gaza", "gazans", "hamas", "hassan", "nasrallah",
    "hezbollah", "houthis", "humanitarian", "crisis", "intifada", "iran",
    "muhammad", "sinuar", "naim qassem", "nakba", "nukhba", "oppressed",
    "organization", "palestine", "palestinians", "plo", "refugees", "resistance",
    "resisters", "sinuar", "terrorists", "tyrants", "victims"
]


# Function to split text into sentences
def split_into_sentences(text):
    if pd.isna(text):
        return []
    # Improved regex for better sentence splitting
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)(?=\s|$)', text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]  # Remove empty sentences


# returns true if the sentence should be taken, else false
# we are taking all the sentences that are "complicated"
def check_sentence(sentence):
    israel = False
    palestine = False
    sentence = sentence.lower()
    for word in israel_words:
        if word.lower() in sentence:
            israel = True
    for word in palestine_words:
        if word.lower() in sentence:
            palestine = True
    if (palestine and israel): #or (not palestine and not israel):
        return True
    return False


# Load the original Excel file
input_file = "posts_first_targil.xlsx"
output_file = "sentences.xlsx"

# Read all sheets into a dictionary of DataFrames
sheets = pd.read_excel(input_file, sheet_name=None)

processed_sheets = {}

# Process each sheet
for sheet_name, df in sheets.items():
    new_rows = []

    # Iterate through each row in the sheet
    for idx, row in df.iterrows():
        newspaper = row.get("Newspaper", "Unknown")
        title = row.get("title", "")
        body_text = row.get("Body Text", "")

        # Split title and body text into sentences

        body_sentences = split_into_sentences(body_text)

        # Add title sentence as sentence 1
        document_number = idx + 1  # Assuming document number corresponds to row index
        sentence_number = 1
        if check_sentence(title.strip())==True:
            new_rows.append({
                "Newspaper": newspaper,
                "Document Number": document_number,
                "Sentence Number": sentence_number,
                "Sentence": title,
                "Label By Hand": ""  # Add empty 'Label By Hand' column
            })
            sentence_number = 2

        # Add body sentences starting from sentence 2
        for sentence in body_sentences:
            if check_sentence(sentence.strip())==True:
                new_rows.append({
                    "Newspaper": newspaper,
                    "Document Number": document_number,
                    "Sentence Number": sentence_number,
                    "Sentence": sentence.strip(),
                    "Label By Hand": ""  # Add empty 'Label By Hand' column
                })
                sentence_number += 1

    # Create a new DataFrame for the processed sheet
    processed_sheets[sheet_name] = pd.DataFrame(new_rows)

# Write the processed data to a new Excel file
with pd.ExcelWriter(output_file) as writer:
    for sheet_name, processed_df in processed_sheets.items():
        processed_df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Processed Excel file saved to {output_file}")




