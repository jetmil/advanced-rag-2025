"""
Детальная отладка поиска - ЧТО именно находит векторный поиск
"""
from rag_knowledge_base import LocalRAG
from pathlib import Path

# Настройки
project_dir = Path(__file__).parent
TEXT_FILE = str(project_dir / "cosmic_texts.txt")
DB_PATH = str(project_dir / "chroma_db_cosmic")

print("="*70)
print("ДЕТАЛЬНАЯ ОТЛАДКА ВЕКТОРНОГО ПОИСКА")
print("="*70)

# Создание RAG
rag = LocalRAG(
    text_file_path=TEXT_FILE,
    db_path=DB_PATH,
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    use_gpu=True
)

# Загрузка базы
from langchain_community.vectorstores import Chroma
rag.vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=rag.embeddings
)

print(f"База загружена: {DB_PATH}")
print(f"Документов в базе: {rag.vectorstore._collection.count()}")
print()

# Настройка retriever
rag.create_qa_chain(retriever_k=10, use_mmr=True)

# ТЕСТ 1: Прямой similarity поиск (без MMR)
print("\n" + "="*70)
print("ТЕСТ 1: Similarity поиск 'Перун'")
print("="*70)

results = rag.vectorstore.similarity_search("Перун", k=5)
print(f"Найдено: {len(results)} документов\n")

for i, doc in enumerate(results, 1):
    content = doc.page_content
    print(f"[{i}] Длина: {len(content)} символов")

    # Ищем слово Перун
    if "Перун" in content:
        idx = content.index("Перун")
        snippet = content[max(0, idx-80):idx+80]
        print(f"    НАЙДЕН 'Перун': ...{snippet}...")
    elif "перун" in content.lower():
        idx = content.lower().index("перун")
        snippet = content[max(0, idx-80):idx+80]
        print(f"    НАЙДЕН 'перун': ...{snippet}...")
    else:
        print(f"    НЕ НАЙДЕН 'Перун' в документе!")
        print(f"    Начало: {content[:200]}")
    print()

# ТЕСТ 2: Поиск с MMR
print("\n" + "="*70)
print("ТЕСТ 2: MMR поиск 'Перун'")
print("="*70)

results_mmr = rag.vectorstore.max_marginal_relevance_search(
    "Перун",
    k=5,
    fetch_k=15,
    lambda_mult=0.5
)
print(f"Найдено: {len(results_mmr)} документов\n")

for i, doc in enumerate(results_mmr, 1):
    content = doc.page_content
    print(f"[{i}] Длина: {len(content)} символов")

    if "Перун" in content:
        idx = content.index("Перун")
        snippet = content[max(0, idx-80):idx+80]
        print(f"    НАЙДЕН 'Перун': ...{snippet}...")
    elif "перун" in content.lower():
        idx = content.lower().index("перун")
        snippet = content[max(0, idx-80):idx+80]
        print(f"    НАЙДЕН 'перун': ...{snippet}...")
    else:
        print(f"    НЕ НАЙДЕН 'Перун' в документе!")
        print(f"    Начало: {content[:200]}")
    print()

# ТЕСТ 3: Проверка embeddings
print("\n" + "="*70)
print("ТЕСТ 3: Поиск других терминов")
print("="*70)

test_terms = ["Фираст", "Пирва", "Анаконда", "космоэнергетика"]
for term in test_terms:
    results = rag.vectorstore.similarity_search(term, k=3)
    found = any(term in doc.page_content or term.lower() in doc.page_content.lower()
                for doc in results)
    status = "НАЙДЕН" if found else "НЕ НАЙДЕН"
    print(f"{term}: {status} (проверено {len(results)} документов)")

print("\n" + "="*70)
print("АНАЛИЗ ЗАВЕРШЕН")
print("="*70)
