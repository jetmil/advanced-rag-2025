"""
Тест поиска конкретных терминов: Перун, Фираст
Проверка логики работы системы
"""
from rag_knowledge_base import LocalRAG
from pathlib import Path
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Настройки
project_dir = Path(__file__).parent
TEXT_FILE = str(project_dir / "cosmic_texts.txt")
DB_PATH = str(project_dir / "chroma_db_test")

logger.info("="*70)
logger.info("ТЕСТ: Проверка поиска терминов Перун, Фираст")
logger.info("="*70)

# Проверка что база существует
import os
if not os.path.exists(DB_PATH):
    logger.error(f"База данных не найдена: {DB_PATH}")
    logger.error("Запустите test_search.py сначала для создания базы")
    exit(1)

logger.info(f"База данных найдена: {DB_PATH}")

# Создание RAG
logger.info("\n1. Загрузка RAG системы...")
rag = LocalRAG(
    text_file_path=TEXT_FILE,
    db_path=DB_PATH,
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    use_gpu=True
)

# Загрузка существующей базы
logger.info("\n2. Загрузка векторной базы...")
from langchain_community.vectorstores import Chroma
rag.vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=rag.embeddings
)
logger.info(f"   База загружена, документов: {rag.vectorstore._collection.count()}")

# Настройка retriever
logger.info("\n3. Настройка retriever с MMR...")
rag.create_qa_chain(retriever_k=10, use_mmr=True)
logger.info(f"   Retriever настроен: k=10, MMR=True")

# Тестовые поиски
logger.info("\n" + "="*70)
logger.info("РЕЗУЛЬТАТЫ ПОИСКА")
logger.info("="*70)

test_queries = [
    "Перун",
    "Фираст",
    "Перун Фираст",  # Оба термина вместе
]

for query in test_queries:
    logger.info(f"\nПоиск: '{query}'")
    logger.info("-" * 70)

    # Поиск релевантных документов
    docs = rag.retriever.get_relevant_documents(query)

    if docs:
        logger.info(f"[OK] Найдено: {len(docs)} документов")

        # Показываем первые 5 результатов
        for i, doc in enumerate(docs[:5], 1):
            content = doc.page_content[:300].replace('\n', ' ')

            # Проверяем наличие термина в тексте
            term_found = False
            for term in ["перун", "фираст", "пирва"]:
                if term in content.lower():
                    logger.info(f"   {i}. [✓ Найден '{term}']: {content}...")
                    term_found = True
                    break

            if not term_found:
                logger.warning(f"   {i}. [⚠ Термин НЕ найден в превью]: {content}...")
    else:
        logger.error("[ERROR] Ничего не найдено!")

logger.info("\n" + "="*70)
logger.info("ТЕСТ ЗАВЕРШЁН")
logger.info("="*70)
