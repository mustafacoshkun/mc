import os
import torch
import time
from collections import deque
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferWindowMemory

def safe_text(text):
    return text.encode("utf-8", "replace").decode("utf-8")

# PDF klasörü ve dosya listesi
pdf_folder = "birlesik_pdfler2/TYT _ Biyoloji (TYT)"
#pdf_files = sorted([f for f in os.listdir(pdf_folder) if f.endswith(".pdf")])
pdf_files = sorted([f.split("_")[4] for f in os.listdir(pdf_folder) if f.endswith(".pdf")])
#print("[✔] PDF dosyaları bulundu:", pdf_files)

# Qdrant istemcisi ve embedding modeli
#client = QdrantClient(path="qdrant_data")
client = QdrantClient(url="http://localhost:6333")
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
print("[✔] Embedding modeli yüklendi.")

# Vector store tanımları
main_vector_store = QdrantVectorStore(
    client=client,
    collection_name="TYT _ Biyoloji (TYT)",
    embedding=embeddings,
)

keywords_vector_store = QdrantVectorStore(
    client=client,
    collection_name="TYT _ Biyoloji (TYT)_keywords",
    embedding=embeddings,
)


# LLM
llm = OllamaLLM(
    model="gemma3:12b",
    temperature=0.1,
    top_p=0.9,
    repeat_penalty=1.1,
    num_predict=200,
    streaming=True
)

""""
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    model="google/gemma-3-1b-it",
    streaming=True
)
"""


# Prompt
qa_prompt = ChatPromptTemplate.from_template("""
Sen bir Öğretmenisin. Aşağıdaki kurallara göre yalnızca verilen "chat_history" (son 3 user–assistant çifti) ve "context" (PDF'den çekilen içerik) içindeki bilgilerle cevap ver:

1. Eğer soru önceki konuşma geçmişine (chat_history) dayalı bir takip sorusuysa:
   a) chat_history veya context içinde doğrudan cevap buluyorsan, eksiksiz şekilde yanıtla.
   b) Bulamazsan "Bu konu hakkında bilgi sahibi değilim…" yaz.

2. Eğer soru yeni bir soru (chat_history ile ilişkisi yoksa):
   a) Context içinde soruya yanıt verebilecek bilgi varsa, oradaki bilgiyi kullanarak cevapla.
   b) Yoksa "Bu konu hakkında bilgi sahibi değilim…" yaz.

Cevabını yalnızca chat_history ve context'e dayandır. Kendi genel bilgin ya da interneti kullanma.
Cevabı Türkçe ver.

—— Konuşma Geçmişi ——  
{chat_history}

—— İçerik ——  
{context}

—— Soru ——  
{question}

Cevap:
""")

# Hafıza başlatma
memory = ConversationBufferWindowMemory(
    k=3,
    memory_key="chat_history",
    return_messages=False,
    human_prefix="Öğrenci",
    ai_prefix="Öğretmen"
)

def get_context_from_pdf(pdf_name, question, k):

    results_semantic = main_vector_store.similarity_search_with_score(
        query=question,
        k=k,
        filter=models.Filter(
            must=[models.FieldCondition(
                key="metadata.source",
                match=models.MatchValue(value=pdf_name)
            )]
        )
    )

    keyword_results = keywords_vector_store.similarity_search_with_score(
        query=question,
        k=k,
        filter=models.Filter(
            must=[models.FieldCondition(
                key="metadata.source",
                match=models.MatchValue(value=pdf_name)
            )]
        )
    )
    chunk_nos = [doc.metadata.get("chunk_no") for doc, _ in keyword_results]

    results_keywords = []
    for chunk_no in chunk_nos:
        chunks = main_vector_store.similarity_search_with_score(
            query="",
            k=2,
            filter=models.Filter(
                must=[models.FieldCondition(
                    key="metadata.chunk_no",
                    match=models.MatchValue(value=chunk_no)
                )]
            )
        )
        results_keywords.extend(chunks)

    # Duplicate chunk_no'lardan kaçınmak için tekilleştirme
    seen_chunk_nos = set()
    unique_docs_with_scores = []
    for doc, score in results_semantic + results_keywords:
        chunk_no = doc.metadata.get("chunk_no")
        if chunk_no not in seen_chunk_nos:
            seen_chunk_nos.add(chunk_no)
            unique_docs_with_scores.append((doc, score))

    # Debug için ekrana yazdırma
    for doc, score in unique_docs_with_scores:
        chunk_no = doc.metadata.get("chunk_no", "Bilinmiyor")
        print(f"* [Chunk #{chunk_no}] [SIM={score:.3f}] {doc.page_content}...\n")


    context_text = "\n---\n".join(safe_text(doc.page_content) for doc, _ in unique_docs_with_scores)
    return context_text

# CLI Döngüsü
def main():
    print("\n[🔍] Sorgulama Başladı. Çıkmak için `q` yazınız.")
    print("[PDF Seçenekleri] :")
    for i, pdf in enumerate(pdf_files):
        print(f"[{i}] {pdf}")


    while True:
        pdf_choice = input("\nPDF dosyasını seçin (0-{}): ".format(len(pdf_files) - 1))
        if pdf_choice.isdigit() and 0 <= int(pdf_choice) < len(pdf_files):
            pdf_name = pdf_files[int(pdf_choice)]
            print("seçilen pdf:", pdf_name)
            break
        else:
            print("❌ Geçersiz seçim.")


    while True:
        k = int(input("Benzer chunk sayısı (0<k<11): "))
        if k > 0 and k < 10:
            break
        else:
            print("❌ Geçersiz k değeri.")

    while True:
        question = input("\n❓ Soru: ").strip()
        if question.lower() == "q":
            print("🚪 Çıkılıyor...")
            break


        context_text = get_context_from_pdf(pdf_name, question, k)
        chat_history = memory.buffer or ""

        print(chat_history)

        formatted = qa_prompt.format(
            chat_history=chat_history,
            context=context_text,
            question=question
        )

        print("🧠 Yanıt:\n")
        response = ""
        for token in llm.stream(formatted):
            print(token, end="", flush=True) # bu şeklide ollama kullanımı
            #print(token.content, end="", flush=True)
            response += safe_text(token) # bu şeklide ollama kullanımı
            #response += safe_text(token.content)
        print("\n")

        memory.save_context(
            {"input": f"Öğrenci: {question}"},
            {"output": f"Öğretmen: {response}"}
        )

if __name__ == "__main__":
    main()
