import re
import json
import string
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertModel, DistilBertTokenizer

# ===============================
# CONFIGURATION
# ===============================
DATA_FILE = "dummy_data.json"
SIMILARITY_THRESHOLD = 0.75

# ===============================
# HELPER FUNCTIONS
# ===============================
def extract_data_from_json(file_path: str):
    """Extract CourseOutcome and ProgramOutcome arrays from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data["CourseOutcome"]), pd.DataFrame(data["ProgramOutcome"])

def preprocess_text(text: str) -> str:
    """Convert text to lowercase and remove punctuation."""
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    return text

def generate_embeddings(text_list, tokenizer, model):
    """Generate DistilBERT embeddings for text."""
    encoded_input = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=128
    )
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = model_output.last_hidden_state.mean(dim=1)
    return embeddings.tolist()

# ===============================
# LOAD DATA FROM dummy_data.ts
# ===============================
co_data, po_data = extract_data_from_json(DATA_FILE)

# Clean descriptions
co_data['cleaned_course_outcome_description'] = co_data['course_outcome_description'].apply(preprocess_text)
po_data['cleaned_program_outcome_description'] = po_data['program_outcome_description'].apply(preprocess_text)

# ===============================
# LOAD BERT MODEL
# ===============================
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# ===============================
# GENERATE EMBEDDINGS
# ===============================
po_data['po_embeddings'] = generate_embeddings(po_data['cleaned_program_outcome_description'].tolist(), tokenizer, model)
co_data['co_embeddings'] = generate_embeddings(co_data['cleaned_course_outcome_description'].tolist(), tokenizer, model)

po_embeddings_array = np.array(po_data['po_embeddings'].tolist())
co_embeddings_array = np.array(co_data['co_embeddings'].tolist())

# ===============================
# CALCULATE SIMILARITIES
# ===============================
similarity_matrix = cosine_similarity(po_embeddings_array, co_embeddings_array)

# ===============================
# MAP RELATIONSHIPS
# ===============================
relationships_df = pd.DataFrame(index=co_data['course_outcome_code'], columns=po_data['program_outcome_code'])

for i in range(similarity_matrix.shape[0]):
    for j in range(similarity_matrix.shape[1]):
        po = po_data.loc[i, 'program_outcome_code']
        co = co_data.loc[j, 'course_outcome_code']
        relationships_df.loc[co, po] = 1 if similarity_matrix[i, j] >= SIMILARITY_THRESHOLD else 0

import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"co_po_matrix_{timestamp}.json"

matrix_dict = relationships_df.to_dict(orient="index")

with open(filename, "w", encoding="utf-8") as f:
    json.dump(matrix_dict, f, indent=4)