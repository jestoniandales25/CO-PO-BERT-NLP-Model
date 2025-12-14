import gradio as gr
import json
import string
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer

SIMILARITY_THRESHOLD = 0.7

# Load BERT once when API starts
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    return text

def generate_embeddings(text_list):
    encoded_input = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    with torch.no_grad():
        output = model(**encoded_input)
    embeddings = output.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

def process_co_po(json_data: dict):
    co_data = pd.DataFrame(json_data["CourseOutcome"])
    po_data = pd.DataFrame(json_data["ProgramOutcome"])

    co_data["cleaned"] = co_data["course_outcome_description"].apply(preprocess_text)
    po_data["cleaned"] = po_data["program_outcome_description"].apply(preprocess_text)

    co_emb = generate_embeddings(co_data["cleaned"].tolist())
    po_emb = generate_embeddings(po_data["cleaned"].tolist())

    similarity_matrix = cosine_similarity(po_emb, co_emb)

    relationships = pd.DataFrame(
        index=co_data["course_outcome_code"],
        columns=po_data["program_outcome_code"]
    )

    for i in range(len(po_data)):
        for j in range(len(co_data)):
            po_code = po_data.loc[i, "program_outcome_code"]
            co_code = co_data.loc[j, "course_outcome_code"]
            relationships.loc[co_code, po_code] = (
                1 if similarity_matrix[i, j] >= SIMILARITY_THRESHOLD else 0
            )

    result_json = {
        co: {po: int(relationships.loc[co, po]) for po in relationships.columns}
        for co in relationships.index
    }

    return result_json

# Wrapper function for Gradio
def process_json(file_obj):
    with open(file_obj.name, "r") as f:
        json_data = json.load(f)
    return process_co_po(json_data)

ui = gr.Interface(
    fn=process_json,
    inputs=gr.File(label="Upload JSON File"),
    outputs=gr.JSON(label="Processed Output"),
    title="CO-PO NLP Model",
    description="Upload your JSON file and process it using BERT NLP Model."
)

if __name__ == "__main__":
    ui.launch()
