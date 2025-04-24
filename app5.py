from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import warnings

app = Flask(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

model_name = 'emrecan/bert-base-turkish-cased-mean-nli-stsb-tr'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

priority_terms = {
    "video": 1.5, "konu özeti": 1.5, "hazır bulunuşluk testi": 1.5,
    "değerlendirme testi": 1.5, "soru çöz": 1.5, "çıkmış sorular": 1.5
}

stop_words = set(["acaba", "ama", "ancak", "aslında", "az", "bana", "bazı", "belki", "ben", "beni", "biz", "bize", "bizi",
                  "bu", "bunu", "çünkü", "da", "daha", "de", "diye", "en", "gibi", "hem", "hep", "her", "hiç", "için", "ile",
                  "ise", "kadar", "ki", "mı", "mi", "mu", "mü", "ne", "neden", "niçin", "o", "oldu", "olduk", "olmak", "olmaz",
                  "on", "onu", "siz", "sizi", "şu", "ve", "veya", "ya", "yani"])

def remove_stop_words(sentence):
    return " ".join([w for w in sentence.split() if w not in stop_words])

def cosine_similarity_score(embedding1, embedding2):
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

def mean_pooling_weighted(model_output, attention_mask, weights):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    weighted_embeddings = token_embeddings * weights.unsqueeze(-1).expand(token_embeddings.size())
    return torch.sum(weighted_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def calculate_similarities(source_sentence, group_filter):
    source_sentence = remove_stop_words(source_sentence)
    source_sentence = source_sentence.lower().replace("'", "").replace("â", "a").replace("Â", "a").replace("I", "ı") \
        .replace("’", "").replace("(", "").replace(")", "").replace("/", " ") \
        .replace(",", "").replace("\"", "").replace(" - ", " ").replace(" -", " ")

    encoded_input = tokenizer([source_sentence], padding=True, truncation=True, return_tensors='pt')

    weights = torch.ones(encoded_input['input_ids'].shape[1])
    for term, weight in priority_terms.items():
        if term in source_sentence:
            term_tokens = tokenizer(term, return_tensors='pt')['input_ids'][0][1:-1]
            for idx, token_id in enumerate(encoded_input['input_ids'][0]):
                if token_id in term_tokens:
                    weights[idx] = weight

    with torch.no_grad():
        model_output = model(**encoded_input)

    source_embedding = mean_pooling_weighted(model_output, encoded_input['attention_mask'], weights).numpy()

    with open("total_embeddings4.pkl", "rb") as f:
        metadata, embeddings = pickle.load(f)

    filtered_lines = []
    for (group_raw, line), emb in zip(metadata, embeddings):
        group = group_raw.strip().replace('"', '')
        if group_filter.strip().lower() == "9.sınıf":
            if group in ["9.Sınıf", "Ortak"]:
                filtered_lines.append((line, emb))
        else:
            if group in ["Diğer", "Ortak"]:
                filtered_lines.append((line, emb))

    results = []
    for line, emb in filtered_lines:
        score = cosine_similarity_score(source_embedding, emb)
        if score >= 0.3:
            results.append({"score": float(f"{score:.2f}"), "line": line})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:10]

@app.route("/search", methods=['GET'])
def similarity():
    sentence = request.args.get("sentence")
    g = request.args.get("group")

    if not sentence or not g:
        return jsonify({"error": "sentence and group are required"}), 400

    line_numbers = []
    with open("total_embeddings4.pkl", "rb") as f:
        metadata, _ = pickle.load(f)
    if g=="1":
        group='9.Sınıf'
    else:
        group='Diğer'
    results = calculate_similarities(sentence, group)

    for score, matched_line in results:
        for idx, (_, original_line) in enumerate(metadata):
            if matched_line == original_line:
                line_numbers.append(str(idx))
                break
    
    return "lines" + (",".join(line_numbers))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
