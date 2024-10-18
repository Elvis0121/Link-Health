import json
from flask import Flask, request, jsonify
import openai
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data from the JSON file
with open('/Users/chipiro/Documents/work sample assessments/link_health/open.json', 'r') as f:
    benefits_data = json.load(f)

# Create a DataFrame for easy processing
df = pd.DataFrame(benefits_data['benefits'])

# Initialize Flask app
app = Flask(__name__)

# Set your OpenAI API key here
openai.api_key = 'org-5IJMMfySC2KRoZbOjyKg0Vad'

# Precompute the TF-IDF matrix for the benefits descriptions
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['description'])

def find_relevant_benefit(query):
    """
    Find the most relevant benefit based on the query using TF-IDF cosine similarity.
    """
    query_vec = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix)
    best_match_idx = similarity_scores.argmax()
    return df.iloc[best_match_idx].to_dict()

def generate_response(query, context):
    """
    Generate a response using the OpenAI GPT model, given a user query and context.
    """
    prompt = (
        f"You are a helpful assistant providing information on federal benefits. "
        f"User's question: '{query}'\n"
        f"Context: '{context}'\n"
        f"Please provide a clear and concise answer."
    )
    
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=150
    )
    
    return response.choices[0].text.strip()

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('question', '')
    
    # Step 1: Retrieve the relevant benefit
    relevant_benefit = find_relevant_benefit(user_input)
    context = f"{relevant_benefit['name']}: {relevant_benefit['description']}. Eligibility: {relevant_benefit['eligibility']}. Application process: {relevant_benefit['application_process']}."
    
    # Step 2: Generate a response using GPT-3.5 with the retrieved context
    bot_response = generate_response(user_input, context)
    
    return jsonify({
        'user_input': user_input,
        'response': bot_response,
        'context': context
    })




from flask import Flask
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

@app.route('/api', methods=['GET'])
@app.route('/home/')
def index():
    return "Hello, World!"

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000)


    # Debug/Development
    #app.run(debug=True, host="0.0.0.0", port="5000")
    # Production
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever(0.25)