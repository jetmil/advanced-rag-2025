"""
Проверка: что именно находится и что отправляется в LLM
"""
from rag_hybrid_search import HybridRAG
from pathlib import Path

project_dir = Path(__file__).parent
TEXT_FILE = str(project_dir / "cosmic_texts.txt")
DB_PATH = str(project_dir / "chroma_db_cosmic")

print("="*70)
print("ПРОВЕРКА КОНТЕКСТА ДЛЯ LLM")
print("="*70)

# Создание Hybrid RAG
rag = HybridRAG(
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

print(f"База загружена: {DB_PATH}\n")

# Поиск документов о "Перун"
question = "Что такое Перун в космоэнергетике?"
print(f"ВОПРОС: {question}")
print("="*70)

docs = rag.hybrid_search(question, k=10)

print("\n" + "="*70)
print("НАЙДЕННЫЕ ДОКУМЕНТЫ (проверка содержимого)")
print("="*70)

for i, doc in enumerate(docs, 1):
    content = doc.page_content
    has_perun = "Перун" in content or "перун" in content.lower()
    status = "ЕСТЬ 'Перун'" if has_perun else "НЕТ 'Перун'"

    print(f"\n[{i}] {status} | Длина: {len(content)} символов")

    if has_perun:
        # Показываем контекст вокруг слова "Перун"
        idx = content.lower().find("перун")
        snippet = content[max(0, idx-150):min(len(content), idx+150)]
        print(f"    Контекст: ...{snippet}...")
    else:
        # Показываем начало документа
        print(f"    Начало: {content[:200]}")

# Формируем контекст
context = "\n\n".join([doc.page_content for doc in docs])

print("\n" + "="*70)
print("ИТОГОВЫЙ КОНТЕКСТ ДЛЯ LLM")
print("="*70)
print(f"Длина контекста: {len(context)} символов")
print(f"Вхождений 'Перун': {context.count('Перун')}")
print(f"Вхождений 'перун' (lowercase): {context.lower().count('перун')}")

if "Перун" in context or "перун" in context.lower():
    print("\nВЫВОД: Контекст СОДЕРЖИТ 'Перун'")
    print("       LLM должен дать ответ!")
    print("\nПервые 1000 символов контекста:")
    print("-"*70)
    print(context[:1000])
else:
    print("\nВЫВОД: Контекст НЕ СОДЕРЖИТ 'Перун'")
    print("       Проблема в гибридном поиске!")
