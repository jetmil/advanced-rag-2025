"""
Гибридный поиск: Векторный + Keyword фильтрация
Решает проблему: embeddings находят похожие документы,
но не обязательно содержащие искомый термин
"""
from rag_knowledge_base import LocalRAG
from pathlib import Path
import re

class HybridRAG(LocalRAG):
    """RAG с гибридным поиском: векторный + keyword"""

    def hybrid_search(self, query: str, k: int = 10, keyword_boost: float = 4.0):
        """
        Гибридный поиск:
        1. Векторный поиск (MMR) - находит семантически похожие документы
        2. Keyword фильтрация - проверяет наличие ключевых слов из запроса
        3. Бустинг - повышает ранг документов с точным совпадением

        Args:
            query: поисковый запрос
            k: количество документов
            keyword_boost: коэффициент усиления для документов с точным совпадением
        """
        # Извлекаем ТОЛЬКО имена собственные (слова с заглавной буквы, >=4 символа)
        # ИСКЛЮЧАЕМ служебные слова
        stopwords = {'Что', 'Как', 'Где', 'Когда', 'Зачем', 'Почему', 'Какой', 'Какая', 'Какие'}
        keywords = []
        words = re.findall(r'\b[А-ЯЁ][а-яё]{3,}\b', query)  # Мин. 4 символа
        keywords = [w for w in words if w not in stopwords]

        print(f"Ключевые термины (имена собственные): {keywords}")

        # 1. ПРЯМОЙ KEYWORD ПОИСК в ChromaDB (если есть имена собственные)
        if keywords:
            # Используем where_document для прямого поиска
            import chromadb
            client = chromadb.PersistentClient(path=self.db_path)
            collection = client.get_collection(name="langchain")

            # Прямой поиск по keyword
            keyword_docs = []
            for keyword in keywords:
                # ChromaDB where_document с $contains
                results = collection.get(
                    where_document={"$contains": keyword},
                    include=["documents", "metadatas"],
                    limit=k * 2  # Берем больше для разнообразия
                )

                # Конвертируем в LangChain Document objects
                from langchain.docstore.document import Document
                for i, doc_text in enumerate(results['documents']):
                    meta = results['metadatas'][i] if results['metadatas'] else {}
                    keyword_docs.append(Document(page_content=doc_text, metadata=meta))

            print(f"Keyword поиск нашел: {len(keyword_docs)} документов с '{keywords}'")

            # Если нашли документы по keyword - используем их
            if keyword_docs:
                vector_docs = keyword_docs
            else:
                # Fallback: векторный поиск если keyword не нашел
                vector_docs = self.vectorstore.max_marginal_relevance_search(
                    query, k=k * 3, fetch_k=k * 9, lambda_mult=0.5
                )
                print(f"Keyword не нашел, fallback на векторный: {len(vector_docs)} документов")
        else:
            # Нет ключевых слов - обычный векторный поиск
            vector_docs = self.vectorstore.max_marginal_relevance_search(
                query, k=k * 3, fetch_k=k * 9, lambda_mult=0.5
            )
            print(f"Векторный поиск нашел: {len(vector_docs)} документов")

        # 2. Keyword фильтрация и ранжирование
        scored_docs = []
        for doc in vector_docs:
            content = doc.page_content
            score = 1.0  # Базовый скор от векторного поиска

            # Подсчет совпадений ключевых слов
            matches = 0
            for keyword in keywords:
                if keyword in content or keyword.lower() in content.lower():
                    matches += 1

            # Бустинг документов с ключевыми словами
            if matches > 0:
                score = score * (keyword_boost ** matches)

            scored_docs.append((score, doc, matches))

        # Сортировка по скору
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Статистика
        docs_with_keywords = sum(1 for _, _, m in scored_docs if m > 0)
        print(f"Документов с ключевыми словами: {docs_with_keywords}/{len(scored_docs)}")

        # Возврат топ-k документов
        result_docs = [doc for _, doc, _ in scored_docs[:k]]

        # Debug info
        print("\nТоп-5 документов:")
        for i, (score, doc, matches) in enumerate(scored_docs[:5], 1):
            preview = doc.page_content[:100].replace('\n', ' ')
            print(f"  [{i}] Score: {score:.2f}, Keywords: {matches}, Preview: {preview}...")

        return result_docs

    def query_hybrid(self, question: str, max_tokens: int = 2000, temperature: float = 0.7) -> dict:
        """Запрос с гибридным поиском"""
        print(f"\n{'='*70}")
        print(f"ГИБРИДНЫЙ ПОИСК: {question}")
        print(f"{'='*70}\n")

        # Гибридный поиск вместо обычного retriever
        relevant_docs = self.hybrid_search(question, k=10)

        # Формирование контекста
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Промпт для русского языка
        prompt = f"""Ты - эксперт по космоэнергетике и эзотерическим практикам.
Используй только информацию из предоставленного контекста для ответа на вопрос.
Если в контексте нет информации для ответа, так и скажи.

Контекст:
{context}

Вопрос: {question}

Подробный ответ:"""

        # Запрос к LM Studio
        try:
            print("\nОтправка запроса к LM Studio...")

            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Ты - эксперт по космоэнергетике, который отвечает на вопросы на основе предоставленной информации."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )

            answer = response.choices[0].message.content
            print(f"Получен ответ (длина: {len(answer)} символов)")

            return {
                "answer": answer,
                "source_documents": relevant_docs,
                "context": context
            }

        except Exception as e:
            return {
                "answer": f"Ошибка при обращении к LM Studio: {str(e)}\n\nПроверьте, что LM Studio запущен и модель загружена!",
                "source_documents": relevant_docs,
                "context": context
            }

def main():
    """Тестирование гибридного поиска"""
    project_dir = Path(__file__).parent
    TEXT_FILE = str(project_dir / "cosmic_texts.txt")
    DB_PATH = str(project_dir / "chroma_db_cosmic")

    print("="*70)
    print("ТЕСТ ГИБРИДНОГО ПОИСКА")
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

    print(f"База загружена: {DB_PATH}")
    print(f"Документов: {rag.vectorstore._collection.count()}")

    # Настройка LLM
    rag.setup_lm_studio_llm(model_name="google/gemma-3-27b")

    # Тест 1: Перун
    print("\n" + "="*70)
    print("ТЕСТ 1: Что такое Перун?")
    print("="*70)

    result = rag.query_hybrid("Что такое Перун в космоэнергетике?")
    print("\n" + "="*70)
    print("ОТВЕТ:")
    print("="*70)
    print(result['answer'])

    # Тест 2: Фираст
    print("\n\n" + "="*70)
    print("ТЕСТ 2: Что такое Фираст?")
    print("="*70)

    result = rag.query_hybrid("Расскажи о частоте Фираст")
    print("\n" + "="*70)
    print("ОТВЕТ:")
    print("="*70)
    print(result['answer'])

if __name__ == "__main__":
    main()
