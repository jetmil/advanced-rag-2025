"""
Инспекция содержимого базы данных
"""
import chromadb
from pathlib import Path

project_dir = Path(__file__).parent
DB_PATH = str(project_dir / "chroma_db_cosmic")

print("="*70)
print("ИНСПЕКЦИЯ БАЗЫ ДАННЫХ")
print("="*70)

# Подключение к ChromaDB
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name="langchain")

print(f"База: {DB_PATH}")
print(f"Документов: {collection.count()}")
print()

# Получаем первые 20 документов
results = collection.get(limit=20, include=["documents", "metadatas"])

print("ПЕРВЫЕ 20 ДОКУМЕНТОВ:")
print("="*70)

for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas']), 1):
    print(f"[{i}] Длина: {len(doc)} символов")
    print(f"    Содержимое: {doc}")
    print(f"    Metadata: {meta}")
    print()

# Ищем документы с "Перун"
print("\n" + "="*70)
print("ПОИСК ДОКУМЕНТОВ С 'Перун'")
print("="*70)

# Проверяем первые 1000 документов
results_large = collection.get(limit=1000, include=["documents"])
count_with_perun = sum(1 for doc in results_large['documents'] if 'Перун' in doc or 'перун' in doc.lower())

print(f"Проверено: {len(results_large['documents'])} документов")
print(f"Содержат 'Перун': {count_with_perun}")

if count_with_perun > 0:
    print("\nПЕРВЫЕ 3 ДОКУМЕНТА С 'Перун':")
    found = 0
    for i, doc in enumerate(results_large['documents']):
        if 'Перун' in doc or 'перун' in doc.lower():
            found += 1
            print(f"\n[{found}] Документ #{i+1}")
            print(f"Длина: {len(doc)} символов")
            idx = doc.lower().find('перун')
            if idx != -1:
                snippet = doc[max(0, idx-100):idx+100]
                print(f"Контекст: ...{snippet}...")
            if found >= 3:
                break
