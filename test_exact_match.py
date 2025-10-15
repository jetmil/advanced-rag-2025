"""
Точный поиск в базе - использую прямую проверку по keywords
"""
import chromadb
from pathlib import Path

project_dir = Path(__file__).parent
DB_PATH = str(project_dir / "chroma_db_cosmic")

print("="*70)
print("ТОЧНЫЙ ПОИСК ПО KEYWORDS (БЕЗ EMBEDDINGS)")
print("="*70)

# Подключение к ChromaDB
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name="langchain")

print(f"База: {DB_PATH}")
print(f"Всего документов: {collection.count()}")
print()

# Поиск всех документов с "Перун" (прямая проверка)
print("Сканирование базы по батчам...")

batch_size = 5000
total = collection.count()
found_docs = []

for offset in range(0, total, batch_size):
    batch = collection.get(
        limit=batch_size,
        offset=offset,
        include=["documents"]
    )

    for i, doc in enumerate(batch['documents']):
        if 'Перун' in doc or 'перун' in doc.lower():
            found_docs.append((offset + i, doc))

    print(f"  Проверено: {min(offset + batch_size, total)}/{total}")

print()
print("="*70)
print(f"НАЙДЕНО ДОКУМЕНТОВ С 'Перун': {len(found_docs)}")
print("="*70)

if found_docs:
    print("\nПервые 10 документов:")
    for i, (doc_id, content) in enumerate(found_docs[:10], 1):
        print(f"\n[{i}] Документ ID: {doc_id}")
        print(f"Длина: {len(content)} символов")
        idx = content.lower().find('перун')
        if idx != -1:
            snippet = content[max(0, idx-100):min(len(content), idx+100)]
            print(f"Контекст: ...{snippet}...")
else:
    print("НИ ОДНОГО ДОКУМЕНТА НЕ НАЙДЕНО!")
    print("Это означает проблему в обработке текста при создании базы.")
