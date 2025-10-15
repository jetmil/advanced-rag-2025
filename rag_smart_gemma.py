"""
SMART RAG Agent 2025 - Gemma3 Function Calling
Умный агент с многоуровневой логикой поиска
"""

import gradio as gr
from rag_advanced_memory import AdvancedRAGMemory
import os
import json
import re
from datetime import datetime
from pathlib import Path
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('smart_agent.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Отключаем debug логи от библиотек (слишком много!)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

# Тот же красивый CSS
MODERN_CSS = """
/* Глобальные стили */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif !important;
}

/* Фон с градиентом - бордово-фиолетовый */
body {
    background: linear-gradient(135deg, #8B0000 0%, #4B0082 50%, #8B008B 100%) !important;
    background-attachment: fixed !important;
    position: relative;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    min-height: 100vh !important;
}

.gradio-container {
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 30px !important;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37) !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    margin: 20px auto !important;
    padding: 30px !important;
    max-width: 95vw !important;
}

/* Карточки */
.gr-box, .gr-form, .gr-panel {
    background: rgba(255, 255, 255, 0.08) !important;
    backdrop-filter: blur(20px) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    padding: 20px !important;
}

/* Инпуты */
.gr-input, .gr-textbox, textarea, input {
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 15px !important;
    color: white !important;
    padding: 12px 16px !important;
}

/* Кнопки */
.gr-button {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0.1)) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 15px !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    transition: all 0.3s !important;
}

.gr-button:hover {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.3), rgba(255, 255, 255, 0.2)) !important;
    transform: translateY(-2px);
}

.gr-button-primary {
    background: linear-gradient(135deg, #8B0000 0%, #8B008B 100%) !important;
    border: none !important;
}

/* Лейблы */
label, .gr-label {
    color: rgba(255, 255, 255, 0.9) !important;
    font-weight: 600 !important;
}

/* Markdown */
.markdown-text, .gr-markdown {
    color: rgba(255, 255, 255, 0.95) !important;
}
"""


class SmartQwenAgent:
    """
    Умный агент на базе Gemma3 с function calling
    Автоматически управляет поиском через RAG и GREP
    """

    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.ULTIMATE_DB_PATH = self.project_dir / "chroma_db_ultimate"
        self.DEFAULT_TEXT_FILE = str(self.project_dir / "cosmic_texts.txt")
        self.EMBEDDING_MODEL = "intfloat/multilingual-e5-large"  # Ultimate модель

        self.rag = None
        self.is_initialized = False
        self.conversation_history = []  # Только финальные ответы!

        # Инструменты доступные для Gemma3
        self.tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": "grep_search",
                    "description": "Точный текстовый поиск в базе знаний. Используй для поиска конкретных имен каналов, терминов, частот. Поддерживает нечёткий поиск (fuzzy) - находит слова даже с пробелами внутри (например 'Мектабу' найдет 'Мект абу').",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Поисковый запрос (имя канала, термин)"
                            },
                            "context_lines": {
                                "type": "integer",
                                "description": "Количество строк контекста до и после (default: 5)",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "rag_semantic_search",
                    "description": "Семантический векторный поиск. Используй для концептуальных вопросов, поиска по смыслу (не точным словам). Хорошо для вопросов типа 'как работает...', 'для чего используется...', 'какие каналы подходят для...'",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Семантический запрос"
                            },
                            "num_sources": {
                                "type": "integer",
                                "description": "Количество документов для поиска (1-100, default: 20)",
                                "default": 20
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "expand_query",
                    "description": "Генерация синонимов и связанных терминов. Используй если подозреваешь что термин может быть написан по-разному в базе (опечатки, варианты написания).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "term": {
                                "type": "string",
                                "description": "Термин для расширения"
                            }
                        },
                        "required": ["term"]
                    }
                }
            }
        ]

    def auto_load_ultimate_db(self, progress=gr.Progress()):
        """Автоматическая загрузка ultimate базы при старте"""
        logger.info("="*70)
        logger.info("AUTO-LOADING ULTIMATE DATABASE")

        try:
            if not self.ULTIMATE_DB_PATH.exists():
                return "❌ Ultimate база не найдена! Создайте её через create_ultimate_db.py"

            progress(0.1, desc="🚀 Загрузка Ultimate базы...")
            logger.info(f"Loading from: {self.ULTIMATE_DB_PATH}")

            # Создаем RAG с ultimate настройками
            self.rag = AdvancedRAGMemory(
                text_file_path=self.DEFAULT_TEXT_FILE,
                db_path=str(self.ULTIMATE_DB_PATH),
                embedding_model=self.EMBEDDING_MODEL,
                max_short_memory=10,  # Увеличено для умного агента
                max_context_tokens=20000,  # Увеличено до 20000 токенов
                summarize_threshold=14000,  # 70% от 20000
                enable_auto_summarize=True,
                use_gpu=True
            )

            progress(0.4, desc="🧠 Загрузка embedding модели (2.2GB)...")

            from langchain_community.vectorstores import Chroma
            self.rag.vectorstore = Chroma(
                persist_directory=str(self.ULTIMATE_DB_PATH),
                embedding_function=self.rag.embeddings
            )

            progress(0.7, desc="🔗 Подключение к Gemma3...")
            # Подключаемся к Gemma3 через LM Studio
            self.rag.setup_lm_studio_llm(model_name="google/gemma-3-27b")

            progress(0.9, desc="⚙️ Настройка retriever...")
            self.rag.create_qa_chain(retriever_k=20, use_mmr=True)

            self.is_initialized = True
            progress(1.0, desc="✅ Готово!")

            logger.info("✅ Ultimate база загружена успешно!")
            logger.info("="*70)

            return f"""✅ SMART Agent готов к работе!

🗄️ База: Ultimate (intfloat/multilingual-e5-large)
🧠 Модель: Gemma 3-27B (function calling)
💾 Память: 10 последних + автосуммаризация
🎯 Контекст: 20000 токенов (увеличенный!)
🔄 Итераций: до 15 (принудительная остановка на 10-й)
💪 Ресурсы: Мощный сервер - больше итераций!

🤖 Gemma3 сам решит какие инструменты использовать!
⚡ Оптимизированная стратегия поиска - меньше зацикливания!"""

        except Exception as e:
            logger.error(f"❌ Ошибка загрузки: {str(e)}", exc_info=True)
            return f"❌ Ошибка: {str(e)}"

    def unload_database(self):
        """Выгрузить базу данных для использования других функций"""
        try:
            if self.rag:
                del self.rag
                self.rag = None

            import gc
            gc.collect()

            self.is_initialized = False
            logger.info("База выгружена")

            return "✅ База выгружена. Теперь можно использовать rag_web_modern.py для создания/загрузки других баз."
        except Exception as e:
            return f"❌ Ошибка выгрузки: {str(e)}"

    def grep_search(self, query: str, context_lines: int = 5):
        """Инструмент: точный текстовый поиск с fuzzy"""
        logger.info(f"[TOOL] grep_search: '{query}'")

        try:
            text_file = self.rag.text_file_path
            with open(text_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Fuzzy поиск по ключевым словам
            stopwords = {'что', 'как', 'где', 'когда', 'зачем', 'почему', 'какой', 'какая', 'какие', 'для', 'работы', 'канал', 'частота'}
            words = re.findall(r'\b[а-яёА-ЯЁ]{3,}\b', query.lower())
            keywords = [w for w in words if w not in stopwords]

            if not keywords:
                pattern = re.compile(re.escape(query), re.IGNORECASE)
            else:
                fuzzy_words = []
                for keyword in keywords[:3]:
                    chars = list(keyword)
                    fuzzy_word = ''.join([re.escape(c) + r'[\s\-]*' for c in chars[:-1]]) + re.escape(chars[-1])
                    fuzzy_words.append(fuzzy_word)
                fuzzy_pattern = '|'.join([f'\\b{fw}\\b' for fw in fuzzy_words])
                pattern = re.compile(fuzzy_pattern, re.IGNORECASE)

            results = []
            for i, line in enumerate(lines):
                if pattern.search(line):
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    context = ''.join(lines[start:end])
                    results.append({
                        'line_num': i + 1,
                        'context': context[:500],  # Ограничиваем для экономии токенов
                        'matched_line': line.strip()[:200]
                    })

                    if len(results) >= 15:  # Максимум 15 результатов
                        break

            logger.info(f"[TOOL] grep_search: найдено {len(results)} совпадений")

            return {
                "found": len(results),
                "results": results,  # Возвращаем ВСЕ результаты (до 15)
                "total": len(results),
                "message": f"Найдено {len(results)} совпадений. Этого достаточно для ответа." if len(results) > 0 else "Ничего не найдено. Попробуй другой поисковый запрос."
            }

        except Exception as e:
            logger.error(f"[TOOL] grep_search error: {e}")
            return {"error": str(e)}

    def rag_semantic_search(self, query: str, num_sources: int = 20):
        """Инструмент: семантический поиск"""
        logger.info(f"[TOOL] rag_semantic_search: '{query}', sources={num_sources}")

        try:
            # Обновляем search_kwargs
            search_kwargs = {
                "k": num_sources,
                "fetch_k": num_sources * 3,
                "lambda_mult": 0.5
            }
            self.rag.retriever.search_kwargs = search_kwargs

            # Получаем документы
            docs = self.rag.retriever.get_relevant_documents(query)

            results = []
            full_docs = []  # Полные документы для анализа
            for doc in docs:
                results.append({
                    "content": doc.page_content[:400],  # Ограничиваем для ответа
                    "metadata": doc.metadata
                })
                full_docs.append(doc.page_content)  # Полный текст для проверки

            # ПРОВЕРКА СООТВЕТСТВИЯ: есть ли в документах информация по запросу
            relevance_check = self._check_topic_relevance(query, full_docs)

            logger.info(f"[TOOL] rag_semantic_search: найдено {len(results)} документов")
            logger.info(f"[TOOL] Проверка соответствия: {relevance_check}")

            return {
                "found": len(results),
                "documents": results,
                "relevance_warning": relevance_check,  # Предупреждение о несоответствии
                "message": f"Найдено {len(results)} релевантных документов. Этого достаточно для качественного ответа!" if len(results) >= 5 else f"Найдено всего {len(results)} документов. Можно попробовать еще один поиск с другими словами, НО ЛУЧШЕ ответить на основе имеющегося."
            }

        except Exception as e:
            logger.error(f"[TOOL] rag_semantic_search error: {e}")
            return {"error": str(e)}

    def _check_topic_relevance(self, query: str, documents: list) -> str:
        """Проверка соответствия темы запроса и найденных документов"""
        query_lower = query.lower()

        # Ключевые слова разных тем
        religious_keywords = ['православ', 'церков', 'богослуж', 'канон', 'литурги', 'молебен', 'собор', 'храм']
        esoteric_keywords = ['космоэнергет', 'канал', 'частот', 'энерги', 'эзотерик', 'магическ', 'обряд', 'ритуал']

        # Проверяем запрос
        is_religious_query = any(kw in query_lower for kw in religious_keywords)

        # Проверяем документы
        doc_text = ' '.join(documents).lower()
        has_esoteric_content = any(kw in doc_text for kw in esoteric_keywords)
        has_religious_content = any(kw in doc_text for kw in religious_keywords)

        # Формируем предупреждение
        if is_religious_query and has_esoteric_content and not has_religious_content:
            return "⚠️ ВНИМАНИЕ: Вопрос про религиозные практики, но найдены ЭЗОТЕРИЧЕСКИЕ материалы. Обязательно укажи это в ответе!"
        elif is_religious_query and not has_esoteric_content and not has_religious_content:
            return "⚠️ ВНИМАНИЕ: В базе НЕТ информации по этой религиозной теме. Скажи об этом честно!"

        return "OK"  # Соответствие нормальное

    def expand_query(self, term: str):
        """Инструмент: генерация синонимов"""
        logger.info(f"[TOOL] expand_query: '{term}'")

        # Простая логика расширения для русских терминов
        term_lower = term.lower()

        # Типичные варианты написания
        variants = [term]

        # Убираем/добавляем дефисы
        if '-' in term:
            variants.append(term.replace('-', ''))
            variants.append(term.replace('-', ' '))

        # Варианты окончаний (Мектабу → Мектаба, Мектаб)
        if term_lower.endswith('у'):
            variants.append(term[:-1] + 'а')
            variants.append(term[:-1])

        logger.info(f"[TOOL] expand_query: варианты {variants}")

        return {
            "original": term,
            "variants": list(set(variants))
        }

    def ask_smart_question(self, question: str, progress=gr.Progress()):
        """
        Умный вопрос с Gemma3 function calling
        """
        if not self.is_initialized:
            return "❌ Система не инициализирована!", "", ""

        if not question.strip():
            return "❌ Введите вопрос!", "", ""

        logger.info("="*70)
        logger.info(f"SMART QUESTION: {question}")

        try:
            # Системный промпт для Gemma3 с защитой от галлюцинаций
            system_prompt = """Ты - ассистент работающий с базой знаний по ЭЗОТЕРИКЕ И КОСМОЭНЕРГЕТИКЕ.

⚠️ СОДЕРЖАНИЕ БАЗЫ ДАННЫХ:
- Космоэнергетические каналы (Фираст, Зевс, Анаконда, Шаон и др.)
- Эзотерические практики и обряды
- Магические ритуалы и заговоры
- Работа с энергиями
- НЕТ информации о канонических религиозных практиках!

🎯 ТВОЯ ЗАДАЧА:
1. Проанализировать вопрос пользователя
2. Использовать инструменты поиска (МАКСИМУМ 2-3 раза!)
3. Ответить СТРОГО на основе найденных документов
4. Если информации нет - ЧЕСТНО сказать об этом!

📋 ИНСТРУМЕНТЫ:
- grep_search: точный поиск имен каналов, терминов
- rag_semantic_search: концептуальный поиск по смыслу
- expand_query: варианты написания (использовать РЕДКО)

⚡ СТРАТЕГИЯ ПОИСКА:
1. Вопрос про конкретный канал:
   → rag_semantic_search(название + ключевые слова, num_sources=30)
   → Дать ответ!

2. Концептуальный вопрос:
   → rag_semantic_search(расширенный запрос, num_sources=50)
   → Дать ответ!

3. Если найдено < 5 результатов:
   → Попробовать grep_search ИЛИ другой запрос
   → Максимум 3 вызова инструментов!

🚫 АБСОЛЮТНЫЕ ЗАПРЕТЫ:
1. НЕ ПРИДУМЫВАЙ информацию! Используй ТОЛЬКО найденные документы!
2. НЕ ДОДУМЫВАЙ детали из своих общих знаний!
3. НЕ ИНТЕРПРЕТИРУЙ эзотерику как религиозные практики!
4. Если вопрос НЕ по теме базы - так и скажи!

✅ ПРАВИЛЬНЫЙ ОТВЕТ если информации нет:
"Извините, в базе знаний содержится информация об эзотерических практиках и космоэнергетике.
По вашему запросу '[тема]' информации не найдено.
Могу помочь с вопросами о космоэнергетических каналах или эзотерических практиках."

✅ ПРАВИЛЬНЫЙ ОТВЕТ если тема не совпадает:
"В базе есть информация об эзотерических обрядах, связанных с [тема], но это НЕ канонические [религия] практики.
Вот что я нашел: [информация из документов с указанием что это эзотерика]"

🔥 КРИТИЧЕСКИ ВАЖНО:
- НЕ делай больше 3 вызовов инструментов
- Лучше сказать "информации нет" чем выдумать
- Всегда указывай что информация из базы ЭЗОТЕРИЧЕСКАЯ
- Показывай откуда взята информация (из каких документов)"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]

            tool_calls_history = []
            max_iterations = 15  # Увеличено до 15 итераций (у вас мощный сервер!)
            force_stop_threshold = 10  # Принудительная остановка на 10-й итерации

            progress(0.1, desc="🧠 Gemma3 планирует поиск...")

            for iteration in range(max_iterations):
                logger.info(f"--- Iteration {iteration + 1} ---")

                # ПРИНУДИТЕЛЬНАЯ ОСТАНОВКА если слишком много итераций
                # iteration начинается с 0, поэтому для остановки на 10-й итерации проверяем >= 9
                if iteration >= (force_stop_threshold - 1):
                    logger.warning(f"⚠️ Принудительная остановка на итерации {iteration + 1}")
                    logger.warning(f"Найдено инструментов: {len(tool_calls_history)}")

                    # Добавляем системное сообщение требующее финального ответа
                    messages.append({
                        "role": "system",
                        "content": f"ВНИМАНИЕ! Это итерация {iteration + 1} из {max_iterations}. У тебя уже есть результаты {len(tool_calls_history)} вызовов инструментов. НЕМЕДЛЕННО дай финальный ответ на основе имеющейся информации. НЕ вызывай больше инструментов!"
                    })

                # Запрос к Gemma3
                response = self.rag.llm_client.chat.completions.create(
                    model="google/gemma-3-27b",
                    messages=messages,
                    tools=self.tools_schema,
                    tool_choice="none" if iteration >= (force_stop_threshold - 1) else "auto",  # Блокируем инструменты на пороге
                    temperature=0.3,  # Низкая для точности
                    max_tokens=4000
                )

                assistant_message = response.choices[0].message

                # Gemma3 хочет вызвать инструменты?
                if assistant_message.tool_calls:
                    progress(0.3 + iteration * 0.1, desc=f"🔧 Выполнение инструментов ({iteration + 1})...")

                    # Добавляем сообщение ассистента
                    messages.append({
                        "role": "assistant",
                        "content": assistant_message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            }
                            for tc in assistant_message.tool_calls
                        ]
                    })

                    # Выполняем каждый вызов инструмента
                    for tool_call in assistant_message.tool_calls:
                        function_name = tool_call.function.name
                        arguments = json.loads(tool_call.function.arguments)

                        logger.info(f"Calling: {function_name}({arguments})")

                        # Вызываем соответствующий инструмент
                        if function_name == "grep_search":
                            result = self.grep_search(**arguments)
                        elif function_name == "rag_semantic_search":
                            result = self.rag_semantic_search(**arguments)
                        elif function_name == "expand_query":
                            result = self.expand_query(**arguments)
                        else:
                            result = {"error": "Unknown function"}

                        # Записываем в историю
                        tool_calls_history.append({
                            "tool": function_name,
                            "args": arguments,
                            "result": result
                        })

                        # Добавляем результат в сообщения
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result, ensure_ascii=False)
                        })

                    continue  # Следующая итерация

                else:
                    # Gemma3 готов дать финальный ответ
                    progress(0.9, desc="✨ Синтез финального ответа...")

                    final_answer = assistant_message.content

                    # ВАЖНО: Сохраняем в память ТОЛЬКО финальный ответ
                    self.rag.short_memory.append({
                        "question": question,
                        "answer": final_answer,
                        "timestamp": datetime.now().isoformat()
                    })

                    # Собираем использованные документы для показа
                    used_documents = []
                    for tc in tool_calls_history:
                        if tc['tool'] == 'rag_semantic_search' and 'result' in tc:
                            docs = tc['result'].get('documents', [])
                            used_documents.extend(docs[:5])  # Первые 5 документов

                    # Форматируем ответ в HTML с подсветкой документов
                    formatted_answer = self._format_answer_html(final_answer, used_documents, tool_calls_history)

                    # Формируем информацию об использованных инструментах в HTML
                    tools_html = self._format_tools_html(tool_calls_history)

                    memory_stats = self.rag.get_memory_stats()
                    memory_html = f"""<div style='padding: 10px;'>
                    <p><b>💾 Память:</b> {memory_stats['short_memory_count']} диалогов | {memory_stats['long_memory_count']} суммаризированных</p>
                    <p><b>🔧 Инструментов:</b> {len(tool_calls_history)}</p>
                    <p><b>📊 Итераций:</b> {iteration + 1}</p>
                    </div>"""

                    logger.info(f"FINAL ANSWER LENGTH: {len(final_answer)} chars")
                    logger.info("="*70)

                    progress(1.0, desc="✅ Готово!")

                    return formatted_answer, tools_html, memory_html

            # Превышен лимит итераций
            return f"❌ Превышен лимит итераций ({max_iterations}). Попробуйте упростить вопрос или задать более конкретный запрос.", "", ""

        except Exception as e:
            logger.error(f"ERROR: {str(e)}", exc_info=True)
            return f"❌ Ошибка: {str(e)}", "", ""

    def _format_answer_html(self, answer: str, documents: list, tools_history: list) -> str:
        """Форматирование ответа в HTML с показом источников"""
        import html

        # Escape HTML в ответе
        answer_escaped = html.escape(answer).replace('\n', '<br>')

        # Проверка на предупреждения
        warnings = []
        for tool in tools_history:
            if 'result' in tool and 'relevance_warning' in tool['result']:
                warning = tool['result']['relevance_warning']
                if warning != "OK":
                    warnings.append(warning)

        warnings_html = ""
        if warnings:
            warnings_html = f"""
            <div class='warning-box'>
                <b>⚠️ ВНИМАНИЕ:</b><br>
                {"<br>".join(set(warnings))}
            </div>
            """

        # Форматируем документы-источники
        sources_html = ""
        if documents:
            sources_html = "<div style='margin-top: 20px;'><hr style='border: 1px solid rgba(255,255,255,0.2);'><h4>📚 Использованные источники:</h4>"
            for i, doc in enumerate(documents[:5], 1):
                content = html.escape(doc.get('content', '')[:200])
                sources_html += f"""
                <div class='source-doc'>
                    <b>Источник {i}:</b><br>
                    <small>{content}...</small>
                </div>
                """
            sources_html += "</div>"

        return f"""
        <div style='padding: 20px; max-height: 600px; overflow-y: auto;' class='scroll-to-bottom'>
            {warnings_html}
            <div style='line-height: 1.6;'>
                {answer_escaped}
            </div>
            {sources_html}
        </div>
        <script>
            // Автопрокрутка вниз
            setTimeout(() => {{
                const container = document.querySelector('.scroll-to-bottom');
                if (container) {{
                    container.scrollTop = container.scrollHeight;
                }}
            }}, 100);
        </script>
        """

    def _format_tools_html(self, tools_history: list) -> str:
        """Форматирование информации об инструментах в HTML"""
        if not tools_history:
            return "<div style='color: rgba(255,255,255,0.6); padding: 10px;'>Инструменты не использовались</div>"

        tools_html = "<div style='padding: 10px;'>"
        for i, tool in enumerate(tools_history, 1):
            tool_name = tool['tool']
            args = tool['args']
            result = tool.get('result', {})

            # Иконка инструмента
            icon = "🔍" if tool_name == "grep_search" else "🧠" if tool_name == "rag_semantic_search" else "🔄"

            # Результаты
            found = result.get('found', 0) if isinstance(result, dict) else 'N/A'

            args_str = ', '.join([f"{k}={v}" for k, v in args.items()])

            tools_html += f"""
            <div style='margin: 5px 0; padding: 8px; background: rgba(255,255,255,0.05); border-radius: 5px;'>
                <b>{icon} {i}. {tool_name}</b><br>
                <small>Параметры: {args_str}</small><br>
                <small>Найдено: {found}</small>
            </div>
            """

        tools_html += "</div>"
        return tools_html

    def get_memory_stats(self):
        """Статистика памяти"""
        if not self.is_initialized:
            return "❌ Система не инициализирована!"

        stats = self.rag.get_memory_stats()
        return f"""📊 Статистика SMART Agent

🕐 Длительность сессии: {stats['session_duration']}
💬 Всего диалогов: {stats['total_questions']}
📝 В короткой памяти: {stats['short_memory_count']}
📚 В долгой памяти: {stats['long_memory_count']}

💾 База: Ultimate (multilingual-e5-large)
🧠 Модель: Gemma 3-27B
⚙️ Автосуммаризация: {'✅' if stats['auto_summarize_enabled'] else '❌'}"""

    def clear_memory(self, keep_summaries: bool):
        """Очистка памяти"""
        if not self.is_initialized:
            return "❌ Система не инициализирована!"
        return f"✅ {self.rag.clear_memory(keep_summaries=keep_summaries)}"

    def export_history(self):
        """Экспорт истории"""
        if not self.is_initialized:
            return "❌ Система не инициализирована!", None

        filename = f"smart_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = self.project_dir / filename
        result = self.rag.export_conversation(str(filepath))
        return f"✅ {result}", str(filepath)

    def create_interface(self):
        """Создание Gradio интерфейса с HTML и автопрокруткой"""
        with gr.Blocks(css=MODERN_CSS, title="SMART RAG Agent 2025", theme=gr.themes.Soft(), head="""
        <style>
        /* Автопрокрутка для ответа */
        .scroll-to-bottom {
            scroll-behavior: smooth;
        }
        /* Подсветка использованных документов */
        .source-doc {
            background: rgba(139, 0, 139, 0.1);
            border-left: 3px solid #8B008B;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
        }
        /* Предупреждения */
        .warning-box {
            background: rgba(255, 165, 0, 0.2);
            border: 2px solid orange;
            padding: 15px;
            margin: 10px 0;
            border-radius: 10px;
        }
        </style>
        <script>
        // Автопрокрутка к новому ответу
        function scrollToAnswer() {
            setTimeout(() => {
                const answerBox = document.querySelector('[data-testid="textbox"]');
                if (answerBox) {
                    answerBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
            }, 300);
        }
        </script>
        """) as interface:

            gr.HTML("""
            <div style='text-align: center; padding: 20px;'>
                <h1 style='background: linear-gradient(135deg, #8B0000, #8B008B); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3em; margin: 0;'>
                    🧠 SMART RAG Agent 2025
                </h1>
                <h3 style='color: rgba(255,255,255,0.8); margin-top: 10px;'>
                    Gemma3 Function Calling • Защита от галлюцинаций • Проверка соответствия
                </h3>
            </div>
            """)

            # Статус инициализации
            with gr.Row():
                init_status = gr.HTML(
                    value="""<div class='warning-box' style='text-align: center;'>
                    ⏳ <b>Нажмите 'Запустить SMART Agent' для автозагрузки Ultimate базы</b>
                    </div>"""
                )

            with gr.Row():
                init_btn = gr.Button("🚀 Запустить SMART Agent", variant="primary", size="lg")
                unload_btn = gr.Button("📤 Выгрузить базу", size="lg")

            gr.Markdown("---")

            # Чат
            gr.Markdown("### 💬 Умный диалог")

            with gr.Row():
                with gr.Column(scale=2):
                    question_input = gr.Textbox(
                        label="💭 Ваш вопрос",
                        placeholder="Например: 'расскажи про канал Фираст' или 'какие каналы для защиты?'",
                        lines=3
                    )
                    ask_btn = gr.Button("✨ Спросить", variant="primary", size="lg")

                with gr.Column(scale=3):
                    answer_output = gr.HTML(
                        label="🤖 Ответ SMART Agent",
                        value="<div style='padding: 20px; text-align: center; color: rgba(255,255,255,0.6);'>Здесь появится ответ...</div>"
                    )

            with gr.Row():
                tools_output = gr.HTML(
                    label="🔧 Использованные инструменты",
                    value="<div style='color: rgba(255,255,255,0.6);'>Здесь появится информация об инструментах...</div>"
                )
                memory_info = gr.HTML(
                    label="📊 Память",
                    value="<div style='color: rgba(255,255,255,0.6);'>Здесь появится информация о памяти...</div>"
                )

            gr.Markdown("---")

            # Управление памятью
            gr.Markdown("### 📊 Управление памятью")

            with gr.Row():
                stats_btn = gr.Button("📊 Статистика", size="lg")
                clear_btn = gr.Button("🧹 Очистить (сохранить суммарии)", size="lg")
                clear_all_btn = gr.Button("🗑️ Очистить всё", variant="stop", size="lg")

            stats_output = gr.Textbox(label="Статистика", lines=8, interactive=False)

            with gr.Row():
                export_btn = gr.Button("💾 Экспорт истории", size="lg")

            export_status = gr.Textbox(label="Статус экспорта", lines=2)
            export_file = gr.File(label="Скачать")

            gr.HTML("""
            <div style='padding: 20px; background: rgba(255,255,255,0.05); border-radius: 15px; margin-top: 20px;'>
            <h3>💡 Как это работает</h3>

            <h4>🤖 Gemma3 автоматически:</h4>
            <ul style='line-height: 1.8;'>
                <li>🧠 Анализирует ваш вопрос</li>
                <li>🔍 Выбирает нужные инструменты (GREP/RAG)</li>
                <li>🔄 Делает несколько итераций поиска если нужно</li>
                <li>✨ Синтезирует финальный ответ</li>
            </ul>

            <h4>🛡️ Защита от галлюцинаций (NEW!):</h4>
            <ul style='line-height: 1.8;'>
                <li>✅ Система знает что в базе - эзотерика и космоэнергетика</li>
                <li>✅ Автоматическая проверка соответствия темы запроса</li>
                <li>✅ Предупреждения если информация не из той области</li>
                <li>✅ Показ использованных источников</li>
                <li>✅ Честное "информации нет" вместо выдумывания</li>
            </ul>

            <h4>📚 База данных содержит:</h4>
            <ul style='line-height: 1.8;'>
                <li>🔮 Космоэнергетические каналы (Фираст, Зевс, Анаконда...)</li>
                <li>⚡ Эзотерические практики и обряды</li>
                <li>🌟 Магические ритуалы</li>
                <li>❌ НЕТ канонических религиозных практик!</li>
            </ul>

            <h4>🔧 Доступно 3 инструмента:</h4>
            <ol style='line-height: 1.8;'>
                <li><code>grep_search</code> - точный поиск с fuzzy</li>
                <li><code>rag_semantic_search</code> - семантический векторный поиск</li>
                <li><code>expand_query</code> - генерация вариантов написания</li>
            </ol>

            <p style='margin-top: 15px; color: rgba(255,255,255,0.7);'>
            💾 <b>В память сохраняется только финальный ответ</b> (не размышления)
            </p>
            </div>
            """)

            # Events
            init_btn.click(self.auto_load_ultimate_db, outputs=[init_status])
            unload_btn.click(self.unload_database, outputs=[init_status])

            ask_btn.click(
                self.ask_smart_question,
                inputs=[question_input],
                outputs=[answer_output, tools_output, memory_info]
            )
            question_input.submit(
                self.ask_smart_question,
                inputs=[question_input],
                outputs=[answer_output, tools_output, memory_info]
            )

            stats_btn.click(self.get_memory_stats, outputs=[stats_output])
            clear_btn.click(lambda: self.clear_memory(True), outputs=[stats_output])
            clear_all_btn.click(lambda: self.clear_memory(False), outputs=[stats_output])
            export_btn.click(self.export_history, outputs=[export_status, export_file])

        return interface


def main():
    agent = SmartQwenAgent()
    interface = agent.create_interface()

    print("="*70)
    print("🧠 SMART RAG Agent 2025 - Gemma3 Function Calling")
    print("="*70)
    print("✨ Умный многоуровневый поиск с автоматическим выбором инструментов")
    print("🗄️ Автозагрузка Ultimate базы (intfloat/multilingual-e5-large)")
    print("💾 В память сохраняются только финальные ответы")
    print("="*70)

    interface.launch(server_name="0.0.0.0", server_port=7861, share=False, show_error=True)


if __name__ == "__main__":
    main()
