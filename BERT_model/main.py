# main.py
from fastapi import FastAPI, UploadFile, File
import json
from BERT_NLP import process_co_po

app = FastAPI()

@app.post("/analyze")
async def analyze_json(file: UploadFile = File(...)):
    content = await file.read()
    json_data = json.loads(content.decode("utf-8"))

    result = process_co_po(json_data)

    return {
        "status": "success",
        "matrix": result
    }
