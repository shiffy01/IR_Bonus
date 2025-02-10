import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import nltk
from nltk.corpus import stopwords


stop_words = set(stopwords.words('english'))


MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)


file_path = "with_prediction_good.xlsx"
df = pd.read_excel(file_path)

print("Start processing...")

def get_bert_vector(sentence):
    tokens = tokenizer.tokenize(sentence)
    filtered_tokens = [t for t in tokens if t not in stop_words]

    if not filtered_tokens:  # if all the words are stop words
        return None

    inputs = tokenizer(filtered_tokens, return_tensors="pt", is_split_into_words=True)
    outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state.squeeze(0)  # (num_tokens, embedding_size)
    summed_vector = torch.sum(token_embeddings, dim=0)

    return summed_vector.detach().numpy()


if "Sentence" in df.columns:
    df["bert_vector"] = df["Sentence"].astype(str).apply(get_bert_vector)

    output_file = "BERT_file.xlsx"
    df.to_excel(output_file, index=False)
    print(f"Processing complete! Results saved to {output_file}")
else:
    print("No 'Sentence' column found in the Excel file. Processing aborted.")
