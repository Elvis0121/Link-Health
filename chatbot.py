import json
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Data processing
# Load and preprocess the data
def extract_text_from_pdf(pdf_path):
    text_data = []
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            text_data.append({
                "page": page_num + 1,
                "content": text
            })
    return text_data

# Split text into chunks of around 500 tokens (adjust if needed)
def chunk_text(text, max_tokens=500):
    words = text.split()
    return [' '.join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

# Example: Chunk all pages into smaller pieces
pdf_path = '/Users/chipiro/Documents/work sample assessments/link_health/LinkHealth Work Assessment.pdf'
text_data = extract_text_from_pdf(pdf_path)

# Store all chunks
document_chunks = []
for doc in text_data:
    document_chunks.extend(chunk_text(doc['content'], max_tokens=500))

# Save chunks as JSON
with open('federal_benefits_chunks.json', 'w', encoding='utf-8') as json_file:
    json.dump(document_chunks, json_file, ensure_ascii=False, indent=4)




# create embeddings for effiecient retrieval
# Load pre-trained model for embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for each chunk
chunk_embeddings = model.encode(document_chunks)

# Create a FAISS index for efficient retrieval
d = chunk_embeddings.shape[1]  # Dimension of embeddings
index = faiss.IndexFlatL2(d)
index.add(np.array(chunk_embeddings))



# Retrieval and Summarization
from transformers import GPT2Tokenizer, GPT2LMHeadModel, T5Tokenizer, T5ForConditionalGeneration

# Load summarization model (e.g., T5 for summaries)
summarizer_tokenizer = T5Tokenizer.from_pretrained('t5-small')
summarizer_model = T5ForConditionalGeneration.from_pretrained('t5-small')

def summarize_text(text, max_length=150):
    inputs = summarizer_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = summarizer_model.generate(inputs, max_length=max_length, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    return summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Retrieve relevant chunks
def retrieve(query, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    retrieved_chunks = [document_chunks[i] for i in indices[0] if i < len(document_chunks)]
    return retrieved_chunks

# # Combine retrieved chunks and summarize
# def prepare_context(query):
#     retrieved_chunks = retrieve(query)
#     combined_context = " ".join(retrieved_chunks)
#     if len(combined_context.split()) > 500:
#         return summarize_text(combined_context)
#     return combined_context




# Generate response with GPT
# Load GPT-2 or a larger model for generation
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
generator = GPT2LMHeadModel.from_pretrained('gpt2')

# def generate_response(context, query, max_length=200):
#     input_text = context + "\nUser: " + query + "\nBot:"
#     inputs = tokenizer(input_text, return_tensors='pt', truncation=True)
#     outputs = generator.generate(**inputs, max_length=max_length, num_return_sequences=1, top_p = 0.9, top_k = 60, repetition_penalty = 1.2, temperature = .6, eos_token_id=tokenizer.encode("\n")[0] )
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)






def prepare_context(query):
    retrieved_chunks = retrieve(query)
    combined_context = " ".join(retrieved_chunks)
    # Optionally, summarize or refine the context
    summarized_context = summarize_text(combined_context) if len(combined_context.split()) > 500 else combined_context
    
    # Provide only a specific portion of the summarized context
    return ' '.join(summarized_context.split()[:300])  # Limit context length


def trim_response(response):
    # Keep only the first few sentences to avoid trailing random info
    sentences = response.split('. ')
    return '. '.join(sentences[:2])  # Adjust to keep 1-2 sentences

def generate_response(context, query, max_length=500):
    input_text = context + "\nUser: " + query + "\nBot:"
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True)
    outputs = generator.generate(
        **inputs,
        max_length=max_length,
        top_k=50,
        top_p=0.9,
        temperature=0.8,
        repetition_penalty=5.0,
        num_return_sequences=1
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return trim_response(response)


query = "What is the LifeLine program"
response = generate_response("", query)
print("\n\n chatbot says: ", response)