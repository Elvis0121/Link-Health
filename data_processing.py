import numpy as np
import faiss
import fitz # for converting the pdf data into json for training the model
import sentence_transformers
import json


# Process the data for training the LH Bot
pdf_path = "/Users/chipiro/Documents/work sample assessments/link_health/LinkHealth Work Assessment.pdf"
def process_pdf(pdf):
    benefits_data = []
    with fitz.open(pdf) as pdf_file:
        for page in range(len(pdf_file)):
            pg = pdf_file.load_page(page)
            content = pg.get_text("text")
            benefits_data.append({
                "page": page + 1,
                "content": content
            }
            )
    return benefits_data

# get the text by running process_pdf
data = process_pdf(pdf_path)

# make it json
json_data_destination = "federal_data.json"
with open(json_data_destination, "w", encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii = False, indent = 4)

