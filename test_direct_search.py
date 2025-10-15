"""
Прямая проверка векторного поиска - что РЕАЛЬНО находится
"""
from rag_knowledge_base import LocalRAG
from pathlib import Path
import sys

# Настройки
project_dir = Path(__file__).parent
TEXT_FILE = str(project_dir / "cosmic_texts.txt")

# Ищем любую существующую базу
import os
db_dirs = [d for d in project_dir.glob("chroma_db*") if d.is_dir()]
if not db_dirs:
    print("НЕТ БАЗ ДАННЫХ!")
    sys.exit(1)

DB_PATH = str(db_dirs[0])
print(f"Используем базу: {DB_PATH}")
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

# Настройка retriever
rag.create_qa_chain(retriever_k=10, use_mmr=True)

# ТЕСТ: Поиск "Перун"
print("\nПОИСК: 'Перун'")
print("="*70)

docs = rag.retriever.get_relevant_documents("Перун")
print(f"Найдено документов: {len(docs)}\n")

for i, doc in enumerate(docs[:5], 1):
    content = doc.page_content
    print(f"--- Документ {i} (длина: {len(content)} символов) ---")

    # Проверяем наличие "Перун" в тексте
    if "Перун" in content or "перун" in content.lower():
        idx = content.lower().find("перун")
        snippet = content[max(0, idx-200):idx+200]
        print(f"✓ НАЙДЕН 'Перун' в тексте!")
        print(f"Контекст: ...{snippet}...")
    else:
        # Показываем начало документа
        print(f"✗ 'Перун' НЕ НАЙДЕН в этом документе!")
        print(f"Начало: {content[:300]}...")
    print()

print("="*70)
print("ВЫВОД:")
if any("перун" in doc.page_content.lower() for doc in docs[:5]):
    print("✓ Векторный поиск РАБОТАЕТ - находит документы с 'Перун'")
    print("✗ ПРОБЛЕМА в LLM - он игнорирует найденный контекст!")
else:
    print("✗ Векторный поиск НЕ РАБОТАЕТ - не находит 'Перун'")
    print("✗ ПРОБЛЕМА в embeddings или параметрах базы!")
