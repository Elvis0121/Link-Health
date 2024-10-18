import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fitz # reads the pdf file



# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        text_data = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            text_data.append({
                "page": page_num + 1,
                "content": text
            })
    return text_data

# Extract text from a PDF file
pdf_path = '/Users/chipiro/Documents/work sample assessments/link_health/LinkHealth Work Assessment.pdf'
text_data = extract_text_from_pdf(pdf_path)



# Save the extracted text data as JSON
json_path = 'knowledge_base.json'
with open(json_path, 'w', encoding='utf-8') as json_file:
    json.dump(text_data, json_file, ensure_ascii=False, indent=4)

print(f"Extracted data saved to {json_path}")








# Load the data
with open('knowledge_base.json', 'r') as f:
    knowledge_base = json.load(f)

documents = [doc['content'] for doc in knowledge_base]

# Use a pre-trained model to encode the documents
# model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('all-mpnet-base-v2')
document_embeddings = model.encode(documents)

# Create a FAISS index for efficient similarity search
d = document_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(np.array(document_embeddings))


def retrieve(query, top_k=5):
    print("\n\nstart retrieve")
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    retrieved_docs = [documents[i] for i in indices[0]]
    print("\nend retrieve\n")
    return retrieved_docs



from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load a pre-trained GPT model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
generator = GPT2LMHeadModel.from_pretrained('gpt2')

def generate_response(context, query):
    print("start gen", end = "\n\n")
    # Concatenate context with the user query for the generator
    #input_text = context + "\nUser: " + query + "\nBot:"
    input_text = context + query
    inputs = tokenizer(input_text, return_tensors='pt')
    outputs = generator.generate(**inputs, max_length=500, num_return_sequences=1, repetition_penalty = 2.0, temperature = .6, eos_token_id=tokenizer.encode("\n")[0])
    print("end gen", end = "\n\n")
    return tokenizer.decode(outputs[0], skip_special_tokens=True)



def select_relevant_chunks(query, retrieved_docs, max_chunks=3):
    print("\n\nstart chunks")
    query_embedding = model.encode(query)
    chunk_embeddings = model.encode(retrieved_docs)
    scores = np.dot(chunk_embeddings, query_embedding)  # Cosine similarity
    sorted_indices = np.argsort(scores)[::-1]  # Sort by relevance
    selected_chunks = [retrieved_docs[i] for i in sorted_indices[:max_chunks]]
    print("\n \nend chunks")
    return ' '.join(selected_chunks)



from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load a T5 model for summarization
summarizer_tokenizer = T5Tokenizer.from_pretrained('t5-small')
summarizer_model = T5ForConditionalGeneration.from_pretrained('t5-small')

def summarize_text(text, max_length=1500):
    print("\n\nbegin summarize\n")
    inputs = summarizer_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = summarizer_model.generate(inputs, max_length=max_length, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    print("\n\n end summary")
    return summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# def rag_chatbot(query):
#     # Retrieve relevant documents
#     retrieved_docs = retrieve(query)
#     combined_docs = " ".join(retrieved_docs[:5])  # Combine the top 5 retrieved docs
#     summarized_context = summarize_text(combined_docs)
#     response = generate_response(summarized_context, query)
#     return response

def chunk_text(text, max_tokens=200):
    print("mini chunks start")
    # Split text into smaller chunks of approximately `max_tokens` each.
    words = text.split()
    print("end mini chunks")
    return [' '.join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]




def rag_chatbot(query):
    print("\n\nstart rag")
    # Retrieve relevant documents
    retrieved_docs = retrieve(query)
    print("\n\n")
    print("retrieved docs: ", retrieved_docs)
    # Combine the top retrieved docs
    combined_docs = " ".join(retrieved_docs[:5])  # Adjust the number as needed for more context
    print("\n\n combined docs: ", combined_docs)
    # Summarize the combined content to reduce its length
    summarized_context = summarize_text(combined_docs)
    print("\n\n summarized context: ", summarized_context)

    # Further chunk the summarized content if it is still too long
    chunks = chunk_text(summarized_context, max_tokens=150)
    print("\n\nchunks: " ,chunks)
    # Generate responses for each chunk and combine them
    responses = [generate_response(chunk, query) for chunk in chunks]
    print("\n\nresponses: ", responses)
    # Combine responses or select the most relevant one
    final_response = " ".join(responses[:2])  # Adjust the number to limit the response length
    print("\n\n\nfinal response: ", final_response)
    return final_response


# def rag_chatbot(query):
#     # Retrieve relevant documents
#     retrieved_docs = retrieve(query)
#     print("\nretrieved docs in rag_chatbot: ", retrieved_docs)
#     context = "\n".join(retrieved_docs[:3])  # Use top 3 retrieved documents as context
#     response = generate_response(context, query)
#     print("\ncontext: ", context)
#     print("\nresponse: ", response)
#     return response



# user_query = "What is the impact of climate change on polar bears?"
# response = rag_chatbot(user_query)
# print("Bot:", response)

# print()
# print()
query = "Description of Transitional Aid to Families with Dependent Children"
ans = rag_chatbot(query)
print("Response: ", ans)
# print()
# print()



import re

def parse_sections(text):
    # Use regex patterns or simple string parsing to identify sections
    term_pattern = r"(?<=Term:)[^\n]+"  # Assumes 'Term:' precedes the term
    definition_pattern = r"(?<=Definition:)[^\n]+"  # Assumes 'Definition:' precedes the definition
    eligibility_pattern = r"(?<=Eligibility:)[^\n]+"  # Assumes 'Eligibility:' precedes the eligibility details
    
    # Extract terms, definitions, and eligibility using the patterns
    term_match = re.search(term_pattern, text, re.IGNORECASE)
    definition_match = re.search(definition_pattern, text, re.IGNORECASE)
    eligibility_match = re.search(eligibility_pattern, text, re.IGNORECASE)
    
    # Extract content for other general information
    # Assumes that content is everything after 'Content:' if it exists
    content_pattern = r"(?<=Content:)[\s\S]+"  # Extracts everything after 'Content:'
    content_match = re.search(content_pattern, text, re.IGNORECASE)
    
    # Get matched text or set to None if no match is found
    term = term_match.group(0).strip() if term_match else None
    definition = definition_match.group(0).strip() if definition_match else None
    eligibility = eligibility_match.group(0).strip() if eligibility_match else None
    content = content_match.group(0).strip() if content_match else None
    
    return {
        "term": term,
        "definition": definition,
        "eligibility": eligibility,
        "content": content
    }


# parsed_data = []

# # Loop through the text data from each page and extract structured data
# for page_text in text_data:
#     # Parse each page's text to extract the term, definition, eligibility, and content
#     page_data = parse_sections(page_text)
    
#     # Only add if there is valid data (e.g., term and definition are present)
#     if page_data["term"] and page_data["definition"]:
#         parsed_data.append(page_data)


# import json

# json_path = 'federal_benefits_data.json'
# with open(json_path, 'w', encoding='utf-8') as json_file:
#     json.dump(parsed_data, json_file, ensure_ascii=False, indent=4)

# print(f"Data saved to {json_path}")
