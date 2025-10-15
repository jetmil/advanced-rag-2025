"""
Создание ULTIMATE базы данных с лучшей embedding моделью
intfloat/multilingual-e5-large - лучшая для русского языка
"""
from rag_knowledge_base import LocalRAG
from pathlib import Path
import sys

# Настройки
project_dir = Path(__file__).parent
TEXT_FILE = str(project_dir / "cosmic_texts.txt")
DB_PATH = str(project_dir / "chroma_db_ultimate")  # Новая ULTIMATE база

print("="*70)
print("CREATING ULTIMATE DATABASE")
print("="*70)
print(f"Embedding model: intfloat/multilingual-e5-large")
print(f"  [+] Best for Russian language")
print(f"  [+] Size: 2.2 GB")
print(f"  [+] Quality: maximum")
print()
print(f"Text file: {TEXT_FILE}")
print(f"Database: {DB_PATH}")
print()

# Sozdanie RAG
print("[1/4] Loading embedding model (few minutes)...")
print("      Downloading 2.2 GB model...")

rag = LocalRAG(
    text_file_path=TEXT_FILE,
    db_path=DB_PATH,
    embedding_model="intfloat/multilingual-e5-large",  # BEST MODEL
    use_gpu=True
)

print("      [+] Model loaded!")

# Zagruzka i razbienie dokumentov
print("\n[2/4] Loading and splitting documents...")
documents = rag.load_and_split_documents(
    chunk_size=500,  # OPTIMAL for short terms search
    chunk_overlap=100
)

# Sozdanie vektornogo hranilishcha
print("\n[3/4] Creating vector database (10-15 minutes)...")
print("      Processing 142,072 documents with GPU...")
vectorstore = rag.create_vectorstore(documents, force_recreate=True)

# Sozdanie retriever
print("\n[4/4] Setting up retriever with MMR...")
rag.create_qa_chain(retriever_k=10, use_mmr=True)

print("\n" + "="*70)
print("DATABASE CREATED SUCCESSFULLY!")
print("="*70)
print(f"Path: {DB_PATH}")
print(f"Documents: {len(documents)}")
print(f"Chunk size: 500")
print(f"MMR: Enabled")
print(f"Embedding: intfloat/multilingual-e5-large (BEST)")
print()

# Test poiska
print("\n" + "="*70)
print("VECTOR SEARCH TEST")
print("="*70)

test_terms = ["Perun", "Firast", "Pirva"]

for term in test_terms:
    print(f"\nSearch: '{term}'")
    docs = rag.retriever.invoke(term)

    found = sum(1 for doc in docs[:5] if term in doc.page_content or term.lower() in doc.page_content.lower())

    if found > 0:
        print(f"  [+] Found {found}/5 documents with '{term}'")
    else:
        print(f"  [-] Not found documents with '{term}'")

print("\n" + "="*70)
print("ULTIMATE DATABASE READY!")
print("="*70)
print("\nUse it in web interface:")
print("  python rag_web_modern.py")
print()
