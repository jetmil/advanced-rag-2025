"""
Тестовый скрипт для проверки поиска Перуна, Пирвы, Фираста
"""
from rag_knowledge_base import LocalRAG
from pathlib import Path

# Настройки
project_dir = Path(__file__).parent
TEXT_FILE = str(project_dir / "cosmic_texts.txt")
DB_PATH = str(project_dir / "chroma_db_test")

print("="*70)
print("ТЕСТ: Проверка поиска Перуна, Пирвы, Фираста")
print("="*70)

# Создание RAG
print("\n1. Создание RAG системы...")
rag = LocalRAG(
    text_file_path=TEXT_FILE,
    db_path=DB_PATH,
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    use_gpu=True
)

# Загрузка и split с новыми параметрами
print("\n2. Загрузка документа с chunk_size=500...")
documents = rag.load_and_split_documents(
    chunk_size=500,
    chunk_overlap=100
)
print(f"   Создано чанков: {len(documents)}")

# Создание векторной базы
print("\n3. Создание векторной базы (это займет 2-3 минуты)...")
import os
if os.path.exists(DB_PATH):
    print("   База уже существует, загружаем...")
    from langchain_community.vectorstores import Chroma
    rag.vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=rag.embeddings
    )
else:
    print("   Создаём новую базу...")
    rag.create_vectorstore(documents, force_recreate=False)

# Создание QA chain с MMR
print("\n4. Настройка retriever с MMR...")
rag.create_qa_chain(retriever_k=10, use_mmr=True)

# Тестовые поиски
print("\n" + "="*70)
print("РЕЗУЛЬТАТЫ ПОИСКА")
print("="*70)

test_queries = [
    "Перун",
    "Пирва",
    "Фираст",
    "Анаконда",
    "Гиперкосмический"
]

for query in test_queries:
    print(f"\nPoisk: '{query}'")
    print("-" * 70)

    # Поиск релевантных документов
    docs = rag.retriever.get_relevant_documents(query)

    if docs:
        print(f"[OK] Naideno: {len(docs)} dokumentov")

        # Показываем первые 3 результата
        for i, doc in enumerate(docs[:3], 1):
            content = doc.page_content[:200].replace('\n', ' ')
            # Подсвечиваем искомое слово
            if query.lower() in content.lower():
                print(f"   {i}. [NAYDENO V TEKSTE]: {content}...")
            else:
                print(f"   {i}. {content}...")
    else:
        print("[ERROR] Nichego ne naideno!")

print("\n" + "="*70)
print("ТЕСТ ЗАВЕРШЁН")
print("="*70)
