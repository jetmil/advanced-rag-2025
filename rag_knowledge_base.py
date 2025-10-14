"""
RAG Knowledge Base System
Работает с локальными моделями из LM Studio
Использует ChromaDB для векторного хранилища
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from openai import OpenAI

class LocalRAG:
    def __init__(
        self,
        text_file_path: str,
        db_path: str = "chroma_db",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        lm_studio_port: int = 1234,
        use_gpu: bool = True
    ):
        """
        Инициализация RAG системы

        Args:
            text_file_path: путь к текстовому файлу
            db_path: путь для сохранения векторной БД
            embedding_model: модель для embeddings
            lm_studio_port: порт LM Studio (по умолчанию 1234)
            use_gpu: использовать GPU для embeddings
        """
        self.text_file_path = text_file_path
        self.db_path = db_path
        self.lm_studio_port = lm_studio_port

        # Настройка embedding модели
        print(f"Loading embedding model: {embedding_model}...")
        model_kwargs = {'device': 'cuda'} if use_gpu else {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        self.vectorstore = None
        self.qa_chain = None

    def load_and_split_documents(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Загрузка и разбиение документа на чанки"""
        print(f"\nLoading document: {self.text_file_path}")

        # Загрузка текста
        loader = TextLoader(self.text_file_path, encoding='utf-8')
        documents = loader.load()

        print(f"Document loaded. Length: {len(documents[0].page_content)} characters")

        # Разбиение на чанки
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        splits = text_splitter.split_documents(documents)
        print(f"Document split into {len(splits)} chunks")

        return splits

    def create_vectorstore(self, documents: List, force_recreate: bool = False):
        """Создание векторного хранилища"""

        if os.path.exists(self.db_path) and not force_recreate:
            print(f"\nLoading existing vector database from {self.db_path}...")
            self.vectorstore = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings
            )
            print(f"Vector database loaded. Contains {self.vectorstore._collection.count()} documents")
        else:
            print(f"\nCreating new vector database...")
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.db_path
            )
            print(f"Vector database created with {len(documents)} documents")
            print(f"Saved to: {self.db_path}")

        return self.vectorstore

    def setup_lm_studio_llm(self, model_name: str = "google/gemma-3-27b"):
        """
        Настройка LLM через LM Studio API (OpenAI-compatible)

        ВАЖНО: Запустите LM Studio и загрузите модель перед использованием!
        """
        print(f"\nConnecting to LM Studio (port {self.lm_studio_port})...")
        print(f"Model: {model_name}")
        print("Make sure LM Studio is running with the model loaded!")

        # LM Studio использует OpenAI-совместимый API
        self.llm_client = OpenAI(
            base_url=f"http://localhost:{self.lm_studio_port}/v1",
            api_key="not-needed"  # LM Studio не требует ключ
        )

        self.model_name = model_name
        return self.llm_client

    def create_qa_chain(self, retriever_k: int = 4):
        """Создание цепочки для QA"""

        if self.vectorstore is None:
            raise ValueError("Vectorstore not created. Call create_vectorstore() first.")

        print(f"\nSetting up QA chain with retriever (k={retriever_k})...")

        # Создание retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": retriever_k}
        )

        print("QA chain ready!")
        return self.retriever

    def query(self, question: str, max_tokens: int = 2000, temperature: float = 0.7) -> dict:
        """
        Запрос к базе знаний

        Args:
            question: вопрос пользователя
            max_tokens: максимальное количество токенов в ответе
            temperature: температура генерации (0.0-1.0)

        Returns:
            dict с ключами 'answer' и 'source_documents'
        """

        if self.retriever is None:
            raise ValueError("QA chain not created. Call create_qa_chain() first.")

        # Получение релевантных документов
        relevant_docs = self.retriever.get_relevant_documents(question)

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

    def interactive_mode(self):
        """Интерактивный режим для общения с базой знаний"""
        print("\n" + "="*70)
        print("RAG Knowledge Base - Interactive Mode")
        print("="*70)
        print("\nCommands:")
        print("  - Type your question in Russian")
        print("  - 'quit' or 'exit' to exit")
        print("  - 'context' to show last retrieved context")
        print("="*70 + "\n")

        last_result = None

        while True:
            try:
                question = input("\nYour question: ").strip()

                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if question.lower() == 'context' and last_result:
                    print("\n--- Last Retrieved Context ---")
                    print(last_result.get('context', 'No context available'))
                    print("--- End of Context ---\n")
                    continue

                if not question:
                    continue

                print("\nSearching knowledge base...")
                result = self.query(question)

                print("\n" + "="*70)
                print("ANSWER:")
                print("="*70)
                print(result['answer'])
                print("\n" + "="*70)
                print(f"Sources: {len(result['source_documents'])} relevant chunks found")
                print("="*70)

                last_result = result

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")

def main():
    """Основная функция для быстрого запуска"""

    # Настройки
    TEXT_FILE = r"C:\Users\PC\Downloads\consolidated_texts_20251014_235421_cleaned.txt"
    DB_PATH = r"C:\Users\PC\chroma_db_kosmoenergy"

    # Модели для embeddings (выберите одну):
    # - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" - быстрая, ~120MB
    # - "sentence-transformers/LaBSE" - отличная для русского, ~470MB
    # - "intfloat/multilingual-e5-large" - очень хорошая, ~2.2GB
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    print("="*70)
    print("RAG Knowledge Base Setup")
    print("="*70)

    # Создание RAG системы
    rag = LocalRAG(
        text_file_path=TEXT_FILE,
        db_path=DB_PATH,
        embedding_model=EMBEDDING_MODEL,
        use_gpu=True  # RTX 3090!
    )

    # Загрузка и обработка документов
    documents = rag.load_and_split_documents(
        chunk_size=1000,  # размер чанка
        chunk_overlap=200  # перекрытие между чанками
    )

    # Создание векторного хранилища
    vectorstore = rag.create_vectorstore(documents, force_recreate=False)

    # Настройка LLM
    rag.setup_lm_studio_llm(model_name="google/gemma-3-27b")

    # Создание QA chain
    rag.create_qa_chain(retriever_k=4)  # количество релевантных чанков

    print("\n" + "="*70)
    print("Setup complete! Starting interactive mode...")
    print("="*70)

    # Запуск интерактивного режима
    rag.interactive_mode()

if __name__ == "__main__":
    main()
