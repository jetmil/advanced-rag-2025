"""
Test hybrid search in AdvancedRAGMemory
"""
from rag_advanced_memory import AdvancedRAGMemory
from pathlib import Path

project_dir = Path(__file__).parent
TEXT_FILE = str(project_dir / "cosmic_texts.txt")
DB_PATH = str(project_dir / "chroma_db_cosmic")

print("="*70)
print("TEST HYBRID SEARCH IN ADVANCED RAG MEMORY")
print("="*70)

# Initialize AdvancedRAGMemory
print("\n[1/3] Loading AdvancedRAGMemory...")
rag = AdvancedRAGMemory(
    text_file_path=TEXT_FILE,
    db_path=DB_PATH,
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    max_short_memory=5,
    max_context_tokens=6000,
    use_gpu=True
)

# Load existing database
print("\n[2/3] Loading existing database...")
from langchain_community.vectorstores import Chroma
rag.vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=rag.embeddings
)

# Setup LLM
print("\n[3/3] Connecting to LM Studio...")
rag.setup_lm_studio_llm(model_name="google/gemma-3-27b")

# Create retriever (needed for MMR)
rag.create_qa_chain(retriever_k=10, use_mmr=True)

print("\n" + "="*70)
print("TESTING HYBRID SEARCH")
print("="*70)

test_queries = [
    "Что такое Перун в космоэнергетике?",
    "Расскажи о частоте Фираст",
    "Как работает Пирва?"
]

for i, query in enumerate(test_queries, 1):
    print(f"\n[TEST {i}] Query: {query}")
    print("-"*70)

    # Call hybrid_search directly
    docs = rag.hybrid_search(query, k=5)

    print(f"Found {len(docs)} documents\n")

    for j, doc in enumerate(docs[:3], 1):
        content = doc.page_content
        preview = content[:150].replace('\n', ' ')
        print(f"  [{j}] {preview}...")

    print()

print("\n" + "="*70)
print("TESTING FULL QUERY (with memory)")
print("="*70)

# Test full query with LLM
result = rag.query("Что такое Перун в космоэнергетике?", max_tokens=500)

print("\nANSWER:")
print("-"*70)
print(result['answer'][:500])
print("-"*70)

print("\nMEMORY STATS:")
stats = result['memory_stats']
for key, value in stats.items():
    print(f"  {key}: {value}")

print("\n" + "="*70)
print("SUCCESS! Hybrid search is working!")
print("="*70)
