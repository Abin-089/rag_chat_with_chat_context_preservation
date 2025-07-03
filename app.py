from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from pathlib import Path
import os
from pypdf import PdfReader
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

genai.configure(api_key="AIzaSyBsellbfHfzBqRznQF4d9V9Ks7T4NNSgz0")

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = genai.GenerativeModel('gemini-2.0-flash')
embedder = SentenceTransformer('all-MiniLM-L6-v2')

index = None
chunks = []
conversation_history = []

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    out = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        out.append(' '.join(chunk))
        i += chunk_size - overlap
    return out

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global index, chunks, conversation_history
    file = request.files['pdf_file']
    filename = secure_filename(file.filename)
    file_path = Path(UPLOAD_FOLDER) / filename
    file.save(file_path)

    reader = PdfReader(file_path)
    all_text = ""
    for page in reader.pages:
        all_text += page.extract_text() + "\n"

    chunks = chunk_text(all_text)
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    conversation_history = []

    return jsonify({'message': f'Uploaded and indexed {len(chunks)} chunks.'})

@app.route('/ask', methods=['POST'])
def ask():
    global conversation_history
    data = request.get_json()
    query = data['question']
    q_emb = embedder.encode([query])
    D, I = index.search(np.array(q_emb).astype('float32'), k=3)
    context = '\n'.join([chunks[i] for i in I[0]])

    history_str = ""
    for turn in conversation_history:
        history_str += f"User: {turn['query']}\nAssistant: {turn['response']}\n"

    prompt = f"""Use the following context and conversation history to answer the question.

Context:
{context}

Conversation History:
{history_str}
User: {query}
Assistant:"""

    response = model.generate_content(prompt)
    answer = response.text.strip()

    conversation_history.append({'query': query, 'response': answer})

    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
