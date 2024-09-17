from flask import Flask, jsonify, request
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import warnings
import logging
from zemberek import (
    TurkishSentenceNormalizer,
    TurkishMorphology,
)

logging.getLogger("zemberek.morphology.turkish_morphology").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)

model_name = 'emrecan/bert-base-turkish-cased-mean-nli-stsb-tr'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

priority_terms = {
    "video": 1.5,
    "konu özeti": 1.5,
    "hazır bulunuşluk testi": 1.5,
    "değerlendirme testi": 1.5,
    "soru çöz": 1.5,
    "çıkmış sorular": 1.5
}
stop_words = set(["acaba", "altı", "altmış", "ama", "ancak", "arada", "artık", "asla", "aslında", "ayrıca", "az", "bana", "bazen", "bazı", "bazıları", "belki", "ben", "benden", "beni", "benim", "beri", "beş", "bile", "bin", "bir", "birkaç", "birkez", "birşey", "birşeyi", "biz", "bize", "bizden", "bizi", "bizim", "böyle", "böylece", "bu", "buna", "bunda", "bundan", "bunlar", "bunları", "bunların", "bunu", "bunun", "burada", "bütün", "çoğu", "çoğuna", "çoğunu", "çok", "çünkü", "da", "daha", "dahi", "dan", "de", "defa", "diye", "doksan", "dokuz", "dolayı", "dolayısıyla", "dört", "edecek", "eden", "ederek", "edilecek", "ediliyor", "edilmesi", "ediyor", "elli", "en", "etmesi", "etti", "ettiği", "ettiğini", "gibi", "göre", "halen", "hangi", "hatta", "hem", "henüz", "hep", "hepsi", "her", "herhangi", "herkes", "herkese", "herkesi", "herkesin", "hiç", "hiçbir", "için", "iki", "ile", "ilgili", "ise", "işte", "kaç", "kadar", "karşın", "kendi", "kendilerine", "kendini", "kendisi", "kendisine", "kendisini", "kez", "ki", "kim", "kime", "kimi", "kimin", "kimisi", "kimse", "kırk", "mı", "mi", "mu", "mü", "nasıl", "ne", "neden", "nedenle", "nerde", "nerede", "nereye", "niye", "niçin", "o", "olan", "olarak", "oldu", "olduğu", "olduğunu", "olduklarını", "olmadı", "olmadığı", "olmak", "olması", "olmayan", "olmaz", "olsa", "olsun", "olup", "olur", "olursa", "oluyor", "on", "ona", "ondan", "onlar", "onlara", "onlardan", "onları", "onların", "onu", "onun", "orada", "otuz", "oysa", "öyle", "pek", "rağmen", "sadece", "sanki", "sekiz", "seksen", "sen", "senden", "seni", "senin", "siz", "sizden", "sizi", "sizin", "son", "sonra", "şayet", "şekilde", "şey", "şeyden", "şeyi", "şeyler", "şu", "şuna", "şunda", "şundan", "şunlar", "şunları", "şunların", "şunu", "şunun", "tarafından", "trilyon", "tüm", "üç", "üzere", "var", "vardı", "ve", "veya", "ya", "yani", "yapacak", "yapılan", "yapılması", "yapıyor", "yapmak", "yaptı", "yaptığı", "yaptığını", "yaptıkları", "yedi", "yerine", "yetmiş", "yine", "yirmi", "yoksa", "yüz", "zaten"])


def remove_stop_words(sentence):
    words = sentence.split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)
def cosine_similarity_score(embedding1, embedding2):
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

def mean_pooling_weighted(model_output, attention_mask, weights):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    weighted_embeddings = token_embeddings * weights.unsqueeze(-1).expand(token_embeddings.size())
    return torch.sum(weighted_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def calculate_similarities(source_sentence):
    encoded_input = tokenizer([source_sentence], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)

    weights = torch.ones(encoded_input['input_ids'].shape[1])
    for term, weight in priority_terms.items():
        if term in source_sentence:
            term_tokens = tokenizer(term, return_tensors='pt')['input_ids'][0][1:-1]
            for idx, token_id in enumerate(encoded_input['input_ids'][0]):
                if token_id in term_tokens:
                    weights[idx] = weight

    source_embedding = mean_pooling_weighted(model_output, encoded_input['attention_mask'], weights).numpy()

    with open('total_embeddings2.pkl', 'rb') as f:
        lines, embeddings = pickle.load(f)

    return source_embedding, lines, embeddings

def similar(source_sentence):
    morphology = TurkishMorphology.create_with_defaults()
    normalizer = TurkishSentenceNormalizer(morphology)
    source_sentence = normalizer.normalize(source_sentence)
    source_sentence = remove_stop_words(source_sentence)
    source_sentence = source_sentence.replace("'", "").replace("â", "a").replace("Â", "a").replace("I", "ı").replace("’", "").replace("(", "").replace(")", "").replace("/", " ").replace(",", "").replace("\",", "").replace("\""," - ", " ").replace(" -", " ").lower()

    source_embedding, lines, embeddings = calculate_similarities(source_sentence)

    subject_names = set()
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 1:
            subject_names.add(parts[1].lower())

    matching_subjects = [subject for subject in subject_names if subject in source_sentence]

    if matching_subjects:
        subject_indices = [i for i, line in enumerate(lines) if len(line.strip().split()) > 1 and any(
            subject in line.split()[1].lower() for subject in matching_subjects)]
        filtered_embeddings = [embeddings[i] for i in subject_indices]
        filtered_similarities = [cosine_similarity_score(source_embedding, emb) for emb in filtered_embeddings]
        filtered_indices = [subject_indices[i] for i in np.argsort(filtered_similarities)[::-1][:10] if filtered_similarities[i] >= 0.5]
        return filtered_indices

    similarities = [cosine_similarity_score(source_embedding, emb) for emb in embeddings]
    top_10_indices = np.argsort(similarities)[::-1][:10]
    filtered_top_indices = [index for index in top_10_indices if similarities[index] >= 0.5]
    return filtered_top_indices
app = Flask(__name__)

@app.route('/similarity', methods=['GET'])
def similarity():
    source_sentence = request.args.get('source_sentence')
    top_10_indices = similar(source_sentence)
    return ",".join(str(num) for num in top_10_indices)

if __name__ =='__main__':
    app.run(debug=True, port=8080)
