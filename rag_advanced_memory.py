"""
RAG с продвинутой памятью и автосуммаризацией
Оптимизировано под RTX 3090 (24GB VRAM) + 32GB RAM
"""

from rag_knowledge_base import LocalRAG
from typing import List, Dict, Optional
import tiktoken
from datetime import datetime
import re

class AdvancedRAGMemory(LocalRAG):
    """
    RAG с умной памятью и автосуммаризацией

    Параметры оптимизированы под:
    - RTX 3090 (24GB VRAM)
    - 32GB RAM
    - Gemma-27B (context: 8192 tokens)
    """

    def __init__(
        self,
        *args,
        max_short_memory: int = 5,           # Последние N сообщений в полном виде
        max_context_tokens: int = 6000,      # Макс токенов для контекста (с запасом)
        summarize_threshold: int = 4000,     # Порог для суммаризации
        enable_auto_summarize: bool = True,  # Автосуммаризация
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Память
        self.short_memory: List[Dict] = []      # Последние сообщения (полные)
        self.long_memory: List[str] = []        # Суммаризированная история
        self.session_start = datetime.now()

        # Настройки
        self.max_short_memory = max_short_memory
        self.max_context_tokens = max_context_tokens
        self.summarize_threshold = summarize_threshold
        self.enable_auto_summarize = enable_auto_summarize

        # Токенизатор для подсчета
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = None
            print("⚠️ tiktoken недоступен, подсчет токенов приблизительный")

    def _count_tokens(self, text: str) -> int:
        """Подсчет токенов"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Приблизительно: 1 токен ≈ 4 символа для русского
            return len(text) // 4

    def hybrid_search(self, query: str, k: int = 10, keyword_boost: float = 4.0):
        """
        Гибридный поиск: векторный + keyword фильтрация

        1. Векторный поиск (MMR) - находит семантически похожие документы
        2. Keyword фильтрация - проверяет наличие ключевых слов из запроса
        3. Бустинг - повышает ранг документов с точным совпадением

        Args:
            query: поисковый запрос
            k: количество документов
            keyword_boost: коэффициент усиления для документов с точным совпадением
        """
        # Извлекаем ЗНАЧИМЫЕ СЛОВА (русские слова >=4 символа)
        # ИСКЛЮЧАЕМ служебные слова и короткие предлоги
        stopwords = {'что', 'как', 'где', 'когда', 'зачем', 'почему', 'какой', 'какая', 'какие', 'который', 'которая', 'которые', 'этот', 'эта', 'это', 'эти', 'того', 'тому', 'этого', 'общего'}
        keywords = []
        # Ищем ВСЕ русские слова длиной >=4 символа (независимо от регистра)
        words = re.findall(r'\b[а-яёА-ЯЁ]{4,}\b', query.lower())
        keywords = [w.capitalize() for w in words if w not in stopwords]

        # 1. ВЕКТОРНЫЙ ПОИСК (MMR) - всегда используем, так как ChromaDB direct search слишком медленный
        # Увеличиваем k для лучшего охвата при наличии ключевых слов
        if keywords:
            # Больше документов для keyword фильтрации
            vector_docs = self.vectorstore.max_marginal_relevance_search(
                query, k=k * 5, fetch_k=k * 15, lambda_mult=0.3
            )
        else:
            # Обычный векторный поиск
            vector_docs = self.vectorstore.max_marginal_relevance_search(
                query, k=k * 3, fetch_k=k * 9, lambda_mult=0.5
            )

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

        # Возврат топ-k документов
        result_docs = [doc for _, doc, _ in scored_docs[:k]]

        return result_docs

    def _summarize_old_messages(self) -> str:
        """Суммаризация старых сообщений"""
        if len(self.short_memory) < 3:
            return None

        # Берем сообщения для суммаризации (кроме последних 2)
        messages_to_summarize = self.short_memory[:-2]

        if not messages_to_summarize:
            return None

        # Формируем текст для суммаризации
        dialogue = ""
        for msg in messages_to_summarize:
            dialogue += f"Q: {msg['question']}\nA: {msg['answer'][:300]}...\n\n"

        # Запрос на суммаризацию
        summary_prompt = f"""Сделай краткое резюме следующего диалога (2-3 предложения):

{dialogue}

Краткое резюме основных тем и выводов:"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Ты суммаризируешь диалоги кратко и точно."},
                    {"role": "user", "content": summary_prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )

            summary = response.choices[0].message.content

            # Сохраняем в долгую память
            self.long_memory.append(summary)

            # Удаляем суммаризированные сообщения
            self.short_memory = self.short_memory[-2:]

            return summary

        except Exception as e:
            print(f"⚠️ Ошибка суммаризации: {e}")
            return None

    def _format_memory_for_prompt(self, question: str, context: str) -> tuple:
        """
        Формирование промпта с оптимальным использованием памяти
        Возвращает: (prompt, tokens_used)
        """

        # Базовый промпт
        base_prompt = f"""Ты - эксперт по космоэнергетике и эзотерическим практикам.

ВАЖНО: Используй предоставленный контекст из базы знаний как ОСНОВУ для ответа.
Если контекст содержит достаточно информации - опирайся на него в первую очередь.
Но также можешь дополнять ответ своими общими знаниями, если:
- Контекста недостаточно для полного ответа
- Нужно объяснить общие концепции или термины
- Пользователь спрашивает о чем-то за пределами контекста

Текущий вопрос пользователя: {question}

Подробный ответ:"""

        # Подсчитываем токены для базового промпта и контекста
        base_tokens = self._count_tokens(base_prompt)
        context_tokens = self._count_tokens(f"Контекст:\n{context}\n\n")

        available_for_memory = self.max_context_tokens - base_tokens - context_tokens - 500  # запас

        # Формируем историю
        memory_text = ""

        # Добавляем долгую память (суммаризированную)
        if self.long_memory:
            long_mem = "\nПредыдущий контекст разговора:\n" + "\n".join(self.long_memory[-2:]) + "\n"
            long_tokens = self._count_tokens(long_mem)
            if long_tokens < available_for_memory:
                memory_text += long_mem
                available_for_memory -= long_tokens

        # Добавляем короткую память (последние сообщения)
        if self.short_memory:
            recent_msgs = "\nПоследние вопросы:\n"
            for msg in reversed(self.short_memory):
                msg_text = f"Q: {msg['question']}\nA: {msg['answer'][:200]}...\n"
                msg_tokens = self._count_tokens(msg_text)

                if msg_tokens < available_for_memory:
                    recent_msgs = msg_text + recent_msgs
                    available_for_memory -= msg_tokens
                else:
                    break

            if recent_msgs != "\nПоследние вопросы:\n":
                memory_text += recent_msgs

        # Финальный промпт
        final_prompt = f"""Ты - эксперт по космоэнергетике и эзотерическим практикам.

ВАЖНО: Используй предоставленный контекст из базы знаний как ОСНОВУ для ответа.
Если контекст содержит достаточно информации - опирайся на него в первую очередь.
Но также можешь дополнять ответ своими общими знаниями, если:
- Контекста недостаточно для полного ответа
- Нужно объяснить общие концепции или термины
- Пользователь спрашивает о чем-то за пределами контекста

{memory_text}

Контекст из базы знаний:
{context}

Текущий вопрос пользователя: {question}

Подробный ответ:"""

        total_tokens = self._count_tokens(final_prompt)

        return final_prompt, total_tokens

    def query(
        self,
        question: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        force_summarize: bool = False
    ) -> dict:
        """
        Запрос с умной памятью

        Args:
            question: вопрос пользователя
            max_tokens: макс токенов ответа
            temperature: температура генерации
            force_summarize: принудительная суммаризация
        """

        if self.retriever is None:
            raise ValueError("QA chain not created.")

        # Автосуммаризация если нужно
        if self.enable_auto_summarize and len(self.short_memory) >= self.max_short_memory:
            print("🔄 Автосуммаризация памяти...")
            summary = self._summarize_old_messages()
            if summary:
                print(f"✅ Создано резюме: {summary[:100]}...")

        # Принудительная суммаризация
        if force_summarize and self.short_memory:
            print("🔄 Принудительная суммаризация...")
            self._summarize_old_messages()

        # Получение релевантных документов через ГИБРИДНЫЙ ПОИСК
        relevant_docs = self.hybrid_search(question, k=10)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Формирование промпта с оптимальной памятью
        prompt, tokens_used = self._format_memory_for_prompt(question, context)

        # Проверка на превышение лимита
        if tokens_used > self.summarize_threshold and self.enable_auto_summarize:
            print("⚠️ Превышен порог токенов, суммаризация...")
            self._summarize_old_messages()
            # Повторное формирование промпта
            prompt, tokens_used = self._format_memory_for_prompt(question, context)

        # Запрос к LLM
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Ты - эксперт по космоэнергетике с памятью диалога."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )

            answer = response.choices[0].message.content

            # Сохранение в короткую память
            self.short_memory.append({
                "question": question,
                "answer": answer,
                "timestamp": datetime.now().isoformat()
            })

            return {
                "answer": answer,
                "source_documents": relevant_docs,
                "context": context,
                "memory_stats": {
                    "short_memory_size": len(self.short_memory),
                    "long_memory_size": len(self.long_memory),
                    "tokens_used": tokens_used,
                    "tokens_limit": self.max_context_tokens
                }
            }

        except Exception as e:
            return {
                "answer": f"Ошибка: {str(e)}\n\nПроверьте, что LM Studio запущен!",
                "source_documents": relevant_docs,
                "context": context,
                "memory_stats": {
                    "short_memory_size": len(self.short_memory),
                    "long_memory_size": len(self.long_memory),
                    "tokens_used": tokens_used,
                    "tokens_limit": self.max_context_tokens
                }
            }

    def clear_memory(self, keep_summaries: bool = False):
        """Очистка памяти"""
        self.short_memory = []
        if not keep_summaries:
            self.long_memory = []
        return f"Память очищена. Суммарии {'сохранены' if keep_summaries else 'удалены'}."

    def get_memory_stats(self) -> dict:
        """Статистика памяти"""
        return {
            "session_duration": str(datetime.now() - self.session_start),
            "short_memory_count": len(self.short_memory),
            "long_memory_count": len(self.long_memory),
            "total_questions": len(self.short_memory) + sum(1 for _ in self.long_memory),
            "auto_summarize_enabled": self.enable_auto_summarize,
            "max_context_tokens": self.max_context_tokens,
            "summarize_threshold": self.summarize_threshold
        }

    def export_conversation(self, filepath: str):
        """Экспорт всей истории разговора"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Сессия начата: {self.session_start}\n")
            f.write("="*70 + "\n\n")

            # Долгая память
            if self.long_memory:
                f.write("СУММАРИЗИРОВАННАЯ ИСТОРИЯ:\n")
                f.write("-"*70 + "\n")
                for i, summary in enumerate(self.long_memory, 1):
                    f.write(f"{i}. {summary}\n\n")
                f.write("\n")

            # Короткая память
            if self.short_memory:
                f.write("ПОСЛЕДНИЕ СООБЩЕНИЯ:\n")
                f.write("-"*70 + "\n")
                for i, msg in enumerate(self.short_memory, 1):
                    f.write(f"\n[{msg.get('timestamp', 'N/A')}]\n")
                    f.write(f"Вопрос: {msg['question']}\n")
                    f.write(f"Ответ: {msg['answer']}\n")
                    f.write("-"*70 + "\n")

        return f"История сохранена в {filepath}"


# Пример использования
if __name__ == "__main__":
    from pathlib import Path
    project_dir = Path(__file__).parent
    TEXT_FILE = str(project_dir / "cosmic_texts.txt")
    DB_PATH = str(project_dir / "chroma_db_kosmoenergy")

    print("="*70)
    print("RAG с продвинутой памятью")
    print("Оптимизировано для RTX 3090 + 32GB RAM")
    print("="*70)

    rag = AdvancedRAGMemory(
        text_file_path=TEXT_FILE,
        db_path=DB_PATH,
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        max_short_memory=5,          # Последние 5 сообщений
        max_context_tokens=6000,     # Макс 6000 токенов (Gemma-27B context: 8192)
        summarize_threshold=4000,    # Суммаризация при 4000 токенов
        enable_auto_summarize=True,  # Автосуммаризация
        use_gpu=True
    )

    # Загрузка существующей БД
    from langchain_community.vectorstores import Chroma
    print("\n📚 Загрузка векторной базы данных...")
    rag.vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=rag.embeddings
    )

    print("🔗 Подключение к LM Studio...")
    rag.setup_lm_studio_llm(model_name="google/gemma-3-27b")

    print("⚙️ Настройка retriever...")
    rag.create_qa_chain(retriever_k=4)

    print("\n" + "="*70)
    print("✅ Система готова!")
    print("="*70)
    print("\nКоманды:")
    print("  'clear' - очистить память (сохранить суммарии)")
    print("  'clear all' - очистить всё")
    print("  'stats' - статистика памяти")
    print("  'export' - экспорт истории в файл")
    print("  'summarize' - принудительная суммаризация")
    print("  'quit' - выход")
    print("="*70 + "\n")

    while True:
        question = input("\n💬 Ваш вопрос: ").strip()

        if not question:
            continue

        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 До свидания!")
            break

        if question.lower() == 'clear':
            print(rag.clear_memory(keep_summaries=True))
            continue

        if question.lower() == 'clear all':
            print(rag.clear_memory(keep_summaries=False))
            continue

        if question.lower() == 'stats':
            stats = rag.get_memory_stats()
            print("\n📊 Статистика памяти:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            continue

        if question.lower() == 'export':
            filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            filepath = str(project_dir / filename)
            print(rag.export_conversation(filepath))
            continue

        if question.lower() == 'summarize':
            result = rag.query(question, force_summarize=True)
            print("✅ Суммаризация выполнена")
            continue

        # Обычный запрос
        print("\n🔍 Поиск в базе знаний...")
        result = rag.query(question)

        print("\n" + "="*70)
        print("📝 ОТВЕТ:")
        print("="*70)
        print(result['answer'])
        print("\n" + "="*70)

        # Статистика
        stats = result['memory_stats']
        print(f"💾 Память: {stats['short_memory_size']} недавних + {stats['long_memory_size']} суммаризированных")
        print(f"📊 Токены: {stats['tokens_used']}/{stats['tokens_limit']}")
        print("="*70)
