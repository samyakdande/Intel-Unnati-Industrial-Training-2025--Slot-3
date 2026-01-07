import pickle
import time
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

print("🔹 Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large"
)

print("🔹 Loading Chroma DB...")

vectorstore = Chroma(
    collection_name="ncert_multilingual",
    persist_directory="./chroma_ncert_db"
)

data = vectorstore._collection.get(include=["documents"])
documents = data["documents"]

print(f"📄 Total documents/chunks: {len(documents)}")

doc_embeddings = []
start = time.time()

for i, doc in enumerate(documents, 1):
    emb = embeddings.embed_documents([doc])[0]
    doc_embeddings.append(emb)

    if i % 100 == 0:
        elapsed = time.time() - start
        print(f"✅ Embedded {i}/{len(documents)} chunks | {elapsed:.1f}s elapsed")

print("💾 Saving embeddings to disk...")

with open("doc_embeddings.pkl", "wb") as f:
    pickle.dump(doc_embeddings, f)

print("🎉 DONE! Document embeddings saved successfully.")