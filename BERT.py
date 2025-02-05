import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import nltk
from nltk.corpus import stopwords

# הורדת רשימת ה-Stop-Words אם היא לא קיימת
stop_words = set(stopwords.words('english'))

# טעינת המודל וה-tokenizer של BERT
MODEL_NAME = "bert-base-uncased"  # ניתן לשנות למודל אחר במידת הצורך
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)

# קריאת קובץ ה-Excel
file_path = "with_prediction_good.xlsx"  # עדכני עם שם הקובץ שלך
df = pd.read_excel(file_path)  # קריאה של הגיליון היחיד

print("Start processing...")

def get_bert_vector(sentence):
    tokens = tokenizer.tokenize(sentence)
    filtered_tokens = [t for t in tokens if t not in stop_words]

    if not filtered_tokens:  # אם כל המילים במשפט הן Stop-Words
        return None

    inputs = tokenizer(filtered_tokens, return_tensors="pt", is_split_into_words=True)
    outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state.squeeze(0)  # צורת (num_tokens, embedding_size)
    summed_vector = torch.sum(token_embeddings, dim=0)  # סכימה לפי הדרישה

    return summed_vector.detach().numpy()

# בדיקה אם קיימת עמודת "Sentence"
if "Sentence" in df.columns:
    df["bert_vector"] = df["Sentence"].astype(str).apply(get_bert_vector)

    # שמירת התוצאה חזרה לקובץ Excel
    output_file = "BERT_file.xlsx"
    df.to_excel(output_file, index=False)
    print(f"Processing complete! Results saved to {output_file}")
else:
    print("No 'Sentence' column found in the Excel file. Processing aborted.")
