"""
Создание ЧИСТОЙ базы данных с правильными параметрами
"""
from rag_knowledge_base import LocalRAG
from pathlib import Path
import sys

# Настройки
project_dir = Path(__file__).parent
TEXT_FILE = str(project_dir / "cosmic_texts.txt")
DB_PATH = str(project_dir / "chroma_db_labse")  # Новая база с LaBSE

print("="*70)
print("СОЗДАНИЕ НОВОЙ ЧИСТОЙ БАЗЫ ДАННЫХ")
print("="*70)
print(f"Текстовый файл: {TEXT_FILE}")
print(f"База данных: {DB_PATH}")
print()

# Создание RAG
print("[1/4] Инициализация embedding модели...")
print("ИСПОЛЬЗУЕМ: sentence-transformers/LaBSE")
print("(LaBSE специально обучена для русского языка)")
print()
rag = LocalRAG(
    text_file_path=TEXT_FILE,
    db_path=DB_PATH,
    embedding_model="sentence-transformers/LaBSE",  # ЛУЧШАЯ МОДЕЛЬ ДЛЯ РУССКОГО
    use_gpu=True
)

# Загрузка и разбиение документов
print("\n[2/4] Загрузка и разбиение документов...")
documents = rag.load_and_split_documents(
    chunk_size=500,  # ОПТИМАЛЬНО для поиска коротких терминов
    chunk_overlap=100
)

# Создание векторного хранилища
print("\n[3/4] Создание векторной базы данных (это займет несколько минут)...")
vectorstore = rag.create_vectorstore(documents, force_recreate=True)

# Создание retriever
print("\n[4/4] Настройка retriever с MMR...")
rag.create_qa_chain(retriever_k=10, use_mmr=True)

print("\n" + "="*70)
print("БАЗА ДАННЫХ СОЗДАНА УСПЕШНО!")
print("="*70)
print(f"Путь: {DB_PATH}")
print(f"Документов: {len(documents)}")
print(f"Chunk size: 500")
print(f"MMR: Enabled")
print()

# Тест поиска
print("\n" + "="*70)
print("ТЕСТ ПОИСКА: Перун")
print("="*70)

docs = rag.retriever.invoke("Перун")
print(f"Найдено документов: {len(docs)}\n")

found = False
for i, doc in enumerate(docs[:5], 1):
    content = doc.page_content
    if "Перун" in content or "перун" in content.lower():
        found = True
        idx = content.lower().find("перун")
        snippet = content[max(0, idx-100):idx+100]
        print(f"[{i}] НАЙДЕН! ...{snippet}...")
        print()

if found:
    print("="*70)
    print("УСПЕХ! База данных работает корректно!")
    print("="*70)
else:
    print("="*70)
    print("ОШИБКА! Термин не найден!")
    print("="*70)
    sys.exit(1)
