import data_processing # custom module for cleaning the data
import numpy as np
import faiss
import fitz # for converting the pdf data into json for training the model
import sentence_transformers
import json

from transformers import GPT2Tokenizer, GPT2LMHeadModel, T5Tokenizer, T5ForConditionalGeneration
# T5 is for summarization, GPT is for training
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
generator = GPT2LMHeadModel.from_pretrained('gpt2')
summarizer_tokenizer = T5Tokenizer.from_pretrained('t5-small')
summarizer_model = T5ForConditionalGeneration.from_pretrained('t5-small')

pdf_path = "/Users/chipiro/Documents/work sample assessments/link_health/LinkHealth Work Assessment.pdf"

data = data_processing.process_pdf(pdf_path)
# make it json
json_data_destination = "federal_data.json"
with open(json_data_destination, "w", encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii = False, indent = 4)

# load data
all_docs = []
with open("federal_benefits_data.json") as file:
    data = json.load(file)
    all_docs = [doc["content"] for doc in data]

print("here1")
# Pre-trained model encodes all_docs
training_model = sentence_transformers.SentenceTransformer('all-mpnet-base-v2')
embeddings_all_docs = training_model.encode(all_docs)


# similarity search for possible responses using FAISS
data_shape = embeddings_all_docs.shape[1]
index = faiss.IndexFlatL2(data_shape)
index.add(np.array(embeddings_all_docs))

print("here2")
def retrieve(query, top_k = 5):
    embedding_query = training_model.encode([query])
    _,indices = index.search(np.array(embedding_query), top_k)
    return [all_docs[i] for i in indices] # these are the retrieved files


def get_response(context, query):
    text = context + query
    inp = tokenizer(text, return_tensors = "pt")
    out = generator.generate(**inp, max_length = 500, num_return_sequences = 1, repetition_penalty = 1.2, temperature = 0.8, eos_token_id = tokenizer.encode("\n")[0])
    return tokenizer.decode(out[0], skip_special_tokens = True)

def get_chunks(query, retrieved_docs, chunks_limit = 2):
    embeddings_query = training_model.encode(query)
    chunks_embeddings = training_model.encode(retrieved_docs)
    scores = np.dot(chunks_embeddings, embeddings_query) # cosine similarityy
    scores_sorted = np.argsort(scores)[::-1] # get most relevant pieces first
    chunks = [retrieved_docs[i] for i in scores_sorted[:chunks_limit]]
    return " ".join(chunks)

def make_chunks(text, max_tokens = 200):
    words = text.split()
    return [" ".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens)]

def summarize(text, max_length = 1500, min_length = 50):
    inp = summarizer_tokenizer.encode("summarize: " + text, return_tensors = "pt")
    sum_ids = summarizer_model.generate(inp, max_length = max_length, min_length = min_length, length_penalty = 1.2, num_beams = 4, early_stopping = True)
    return summarizer_tokenizer.decode(sum_ids[0], skip_special_tokens = True)


def chat(query):
    retrieved_files = retrieve(query)
    joined_files = " ".join(retrieved_files[:3])
    summary = summarize(joined_files)
    chunks = make_chunks(summary, max_tokens = 100)
    responses = [get_response(chunk, query) for chunk in chunks]
    response_to_user = " ".join(responses[:2])
    return response_to_user

print("here")
query = "Description of Transitional Aid to Families with Dependent Children"
ans = chat(query)
print("ans: ", ans)