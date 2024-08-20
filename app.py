from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def cosine_similarity_score(embedding1, embedding2):
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

def calculate_similarities(source_sentence):
    tokenizer = AutoTokenizer.from_pretrained('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')
    model = AutoModel.from_pretrained('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')

    # Tokenize and encode the source sentence
    encoded_input = tokenizer([source_sentence], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    source_embedding = mean_pooling(model_output, encoded_input['attention_mask']).numpy()

    # Load precomputed embeddings
    with open('precomputed_embeddings.pkl', 'rb') as f:
        lines, embeddings = pickle.load(f)

    # Calculate cosine similarities with precomputed embeddings
    similarities = [cosine_similarity_score(source_embedding, emb) for emb in embeddings]

    return similarities

app = Flask(__name__)

@app.route('/chatbot/<name>', methods=['GET'])
def chatbot(name):
    return jsonify(message=f"Merhaba, {name}!")

@app.route('/chat/<surname>', methods=['GET'])
def chat(surname):
    return jsonify(message=f"JSON alindi, {surname}!",message2=33)

@app.route('/json', methods=['GET','POST'])
def json():
    data2 = request.get_json()
    return data2

@app.route('/similarity', methods=['GET'])
def similarity():
    source_sentence = request.args.get('source_sentence')
    similarities = calculate_similarities(source_sentence)
    top_10_indices = np.argsort(similarities)[::-1][:10]
    return str(top_10_indices)


if __name__ =='__main__':
    app.run(debug=True, port=8080)
