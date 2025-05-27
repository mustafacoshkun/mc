# pdf'yi chuklara ayırma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from uuid import uuid4
from langchain.schema import Document

# --- Stop‐word temizleme fonksiyonu ---
WPT = WordPunctTokenizer()
stop_word_list = stopwords.words('turkish')

#model = SentenceTransformer("dbmdz/bert-base-turkish-cased") # burada modeli kaydettik
#model.save("keybertModels/turkish-sbert")

model = SentenceTransformer("keybertModels/turkish-sbert")  #modeli kayıtlı yerden açıyoruz

# KeyBERT ile kullan
kw_model = KeyBERT(model=model)

output_path = "chunk_log.txt"
#pdf_folder_path = "merged"
pdf_folder_path = "birlesik_pdfler2/TYT _ Biyoloji (TYT)"
#pdf_folder_path = "birlesik_pdfler/Tarih (AYT)"
#pdf_folder_path4 = "birlesik_pdfler/Türk Dili ve Edebiyatı (AYT)"

all_documents = []

keybert_documents = []

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)

def clean_text(text: str) -> str:
    tokens = WPT.tokenize(text.lower())
    filtered = [t for t in tokens if t not in stop_word_list]
    return " ".join(filtered)

global_chunk_no = 1

with open(output_path, "w", encoding="utf-8") as f:
    for filename in os.listdir(pdf_folder_path):
        if not filename.endswith(".pdf"):
            continue
        filename_chuk = filename.split("_")[4]
        print(filename)
        loader = PyPDFLoader(os.path.join(pdf_folder_path, filename))
        docs = loader.load()
        split_docs = splitter.split_documents(docs)

        for chunk in split_docs:
            # 1) Bu chunk için benzersiz numarayı al
            chunk_no = global_chunk_no
            global_chunk_no += 1

            # 2) Metadata’yı güncelle
            chunk.metadata["source"]   = filename_chuk
            chunk.metadata["chunk_no"] = chunk_no

            chunk.page_content = chunk.page_content.strip()

            # 3) KeyBERT ile anahtar kelime+skor çıkarımı
            keywords_with_scores = kw_model.extract_keywords(
                clean_text(chunk.page_content),
                keyphrase_ngram_range=(1, 1),
                top_n=20
            )

            # 4) keybert_documents listesine hem chunk_no hem keywords ekle
            keybert_documents.append({
                "source": filename_chuk,  # ← PDF adı
                "chunk_no": chunk_no,  # ← global sayaç
                "keywords_with_scores": keywords_with_scores
            })
            # 5) Asıl chunk’ı da all_documents’a ekle
            all_documents.append(chunk)

            # 6) Log dosyasına yaz
            f.write(f"=== PDF: {filename_chuk} | Chunk #{chunk_no} ===\n")
            f.write(chunk.page_content.strip() + "\n\n")
#%%
#chunk'ları vektörleştir ve Qdrant'a kaydet
import torch
from uuid import uuid4
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, ScalarQuantizationConfig
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

# === Qdrant Bağlantısı ===
#client = QdrantClient(path="qdrant_data")

client = QdrantClient(url="http://localhost:6333")

collection_name = "TYT _ Biyoloji (TYT)"

if not client.collection_exists(collection_name):
    print(f"Collection '{collection_name}' bulunamadı. Yeni oluşturuluyor...")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        #quantization_config=ScalarQuantizationConfig(always_ram=True),  # RAM'de sakla                    #type=models.ScalarType.INT8
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="embeddingsModels/multilingual-e5-large",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    #embeddings = OllamaEmbeddings(
    #    model="nomic-embed-text:v1.5",
    #)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    chunk_ids = [str(uuid4()) for _ in all_documents]
    vector_store.add_documents(documents=all_documents, ids=chunk_ids)

    print(f"[✔] {len(all_documents)} adet belge vektörleştirildi ve '{collection_name}' koleksiyonuna kaydedildi.")
else:
    print(f"[✔] Zaten '{collection_name}' koleksiyonu mevcut. Embedding işlemi yapılmadı.")
#%%
import torch
from uuid import uuid4
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, ScalarQuantizationConfig
from langchain_huggingface import HuggingFaceEmbeddings

#client = QdrantClient(path="qdrant_data")
# ——— 2) Keyword’ler için ayrı koleksiyon —————————————
keyword_collection = "TYT _ Biyoloji (TYT)_keywords"

# Koleksiyon yoksa oluştur
if not client.collection_exists(keyword_collection):
    print(f"Collection '{keyword_collection}' bulunamadı. Oluşturuluyor...")
    client.create_collection(
        collection_name=keyword_collection,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )
else:
    print(f"[✔] '{keyword_collection}' koleksiyonu zaten mevcut.")

embeddings = HuggingFaceEmbeddings(
    model_name="embeddingsModels/multilingual-e5-large",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Her halükârda keyword_store’u tanımla
keyword_store = QdrantVectorStore(
    client=client,
    collection_name=keyword_collection,
    embedding=embeddings,
)

# keybert_documents -> Document objeleri
keyword_docs = []
for entry in keybert_documents:
    source    = entry["source"]       # PDF adı
    chunk_no  = entry["chunk_no"]
    for kw, score in entry["keywords_with_scores"]:
        keyword_docs.append(
            Document(
                page_content=kw,
                metadata={
                    "source":    source,
                    "chunk_no":  chunk_no,
                    "keyword":   kw,
                    "score":     score
                }
            )
        )

# UUID listeleri
keyword_ids = [str(uuid4()) for _ in keyword_docs]

# Eğer koleksiyon yeni oluşturulduysa ekle, değilse güncelle
# (İlk versiyon için: her zaman baştan ekleyebilirsin)
keyword_store.add_documents(documents=keyword_docs, ids=keyword_ids)
print(f"[✔] {len(keyword_docs)} keyword vektörü '{keyword_collection}' koleksiyonuna kaydedildi.")

#%% sorgu yapma
import torch
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import models
from qdrant_client import QdrantClient

# QdrantClient'ı başlat
client = QdrantClient(path="qdrant_data")

# Koleksiyon isimleri
keywords_collection_name = "biology_collection_keywords"
main_collection_name = "biology_collection"

# Embedding modelini yükle
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# 1) Anahtar kelime koleksiyonuna bağlan
keywords_vector_store = QdrantVectorStore(
    client=client,
    collection_name=keywords_collection_name,
    embedding=embeddings,
)

print("1. Adım: biology_collection_keywords içinde arama yapılıyor...")

# Anahtar kelime koleksiyonunda arama yap
keyword_results = keywords_vector_store.similarity_search_with_score(
    query="kolajen nedir",
    k=2,
    filter=models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.source",
                match=models.MatchValue(value="5.Proteinler.pdf")
            )
        ]
    )
)

# Bulunan chunk_no'ları topla
target_chunk_nos = []
for doc, score in keyword_results:
    src = doc.metadata.get("source", "—")
    chunk_no = doc.metadata.get("chunk_no", "—")
    target_chunk_nos.append(chunk_no)
    print(f"* [SIM={score:.3f}] {doc.page_content} (source={src}, chunk_no={chunk_no})")
    print("*" + "-" * 50)

print(f"Bulunan chunk_no'lar: {target_chunk_nos}")

# 2) Ana koleksiyona bağlan
main_vector_store = QdrantVectorStore(
    client=client,
    collection_name=main_collection_name,
    embedding=embeddings,
)

print("\n\n")

# Her bir chunk için ayrı sorgu yapıp sonuçları birleştir
all_results = []
for chunk_no in target_chunk_nos:
    chunk_results = main_vector_store.similarity_search_with_score(
        query="",  # Boş sorgu
        k=len(target_chunk_nos),       # Her chunk için kaç sonuç
        filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.chunk_no",
                    match=models.MatchValue(value=chunk_no)
                )
            ]
        )
    )
    all_results.extend(chunk_results)


if all_results:
    for doc, score in all_results:
        print(f"Chunk No: {doc.metadata.get('chunk_no')}")
        print(f"Source: {doc.metadata.get('source', '—')}")
        print(f"Content: {doc.page_content}")
        print("-" * 70)
else:
    print(f"Ana koleksiyonda belirtilen chunk numaralarına sahip içerik bulunamadı.")