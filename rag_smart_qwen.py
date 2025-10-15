"""
SMART RAG Agent 2025 - Qwen3 Function Calling
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
    Умный агент на базе Qwen3 с function calling
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

        # Инструменты доступные для Qwen3
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
                max_context_tokens=16000,  # Максимум для RTX 3090
                summarize_threshold=11000,
                enable_auto_summarize=True,
                use_gpu=True
            )

            progress(0.4, desc="🧠 Загрузка embedding модели (2.2GB)...")

            from langchain_community.vectorstores import Chroma
            self.rag.vectorstore = Chroma(
                persist_directory=str(self.ULTIMATE_DB_PATH),
                embedding_function=self.rag.embeddings
            )

            progress(0.7, desc="🔗 Подключение к Qwen3...")
            # Подключаемся к Qwen3 через LM Studio
            self.rag.setup_lm_studio_llm(model_name="qwen/qwen3-30b-a3b-2507")

            progress(0.9, desc="⚙️ Настройка retriever...")
            self.rag.create_qa_chain(retriever_k=20, use_mmr=True)

            self.is_initialized = True
            progress(1.0, desc="✅ Готово!")

            logger.info("✅ Ultimate база загружена успешно!")
            logger.info("="*70)

            return f"""✅ SMART Agent готов к работе!

🗄️ База: Ultimate (intfloat/multilingual-e5-large)
🧠 Модель: Qwen3-30B-A3B (function calling)
💾 Память: 10 последних + автосуммаризация
🎯 Контекст: 16000 токенов (максимум для RTX 3090)

🤖 Qwen3 сам решит какие инструменты использовать!"""

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
                "results": results[:10],  # Возвращаем первые 10
                "total": len(results)
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
            for doc in docs:
                results.append({
                    "content": doc.page_content[:400],  # Ограничиваем
                    "metadata": doc.metadata
                })

            logger.info(f"[TOOL] rag_semantic_search: найдено {len(results)} документов")

            return {
                "found": len(results),
                "documents": results
            }

        except Exception as e:
            logger.error(f"[TOOL] rag_semantic_search error: {e}")
            return {"error": str(e)}

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
        Умный вопрос с Qwen3 function calling
        """
        if not self.is_initialized:
            return "❌ Система не инициализирована!", "", ""

        if not question.strip():
            return "❌ Введите вопрос!", "", ""

        logger.info("="*70)
        logger.info(f"SMART QUESTION: {question}")

        try:
            # Системный промпт для Qwen3
            system_prompt = """Ты - экспертный ассистент по космоэнергетике с доступом к инструментам поиска.

Твоя задача:
1. Проанализировать вопрос пользователя
2. Решить какие инструменты нужны (grep_search, rag_semantic_search, expand_query)
3. Вызвать инструменты (можно несколько раз если нужно)
4. Синтезировать финальный ответ на основе найденной информации

Правила работы с инструментами:
- grep_search: для поиска конкретных имен каналов, терминов (точный поиск)
- rag_semantic_search: для концептуальных вопросов, поиска по смыслу
- expand_query: если подозреваешь опечатки или варианты написания

Стратегия:
1. Если вопрос про конкретный канал - сначала grep_search(имя канала)
2. Если нужен контекст - rag_semantic_search(концептуальный запрос)
3. Если мало информации - сделай дополнительный поиск
4. Финальный ответ давай подробный, со ссылками на источники

ВАЖНО: Всегда проверяй достаточно ли информации перед финальным ответом!"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]

            tool_calls_history = []
            max_iterations = 7  # Максимум 7 итераций

            progress(0.1, desc="🧠 Qwen3 планирует поиск...")

            for iteration in range(max_iterations):
                logger.info(f"--- Iteration {iteration + 1} ---")

                # Запрос к Qwen3
                response = self.rag.llm_client.chat.completions.create(
                    model="qwen/qwen3-30b-a3b-2507",
                    messages=messages,
                    tools=self.tools_schema,
                    tool_choice="auto",
                    temperature=0.3,  # Низкая для точности
                    max_tokens=4000
                )

                assistant_message = response.choices[0].message

                # Qwen3 хочет вызвать инструменты?
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
                    # Qwen3 готов дать финальный ответ
                    progress(0.9, desc="✨ Синтез финального ответа...")

                    final_answer = assistant_message.content

                    # ВАЖНО: Сохраняем в память ТОЛЬКО финальный ответ
                    self.rag.add_to_memory(question, final_answer)

                    # Формируем информацию об использованных инструментах
                    tools_used = "\n".join([
                        f"🔧 {i+1}. {tc['tool']}({', '.join([f'{k}={v}' for k, v in tc['args'].items()])})"
                        for i, tc in enumerate(tool_calls_history)
                    ])

                    memory_stats = self.rag.get_memory_stats()
                    memory_info = f"""💾 Память: {memory_stats['short_memory_count']} диалогов | {memory_stats['long_memory_count']} суммаризированных
🔧 Использовано инструментов: {len(tool_calls_history)}
📊 Итераций: {iteration + 1}"""

                    logger.info(f"FINAL ANSWER LENGTH: {len(final_answer)} chars")
                    logger.info("="*70)

                    progress(1.0, desc="✅ Готово!")

                    return final_answer, tools_used, memory_info

            # Превышен лимит итераций
            return "❌ Превышен лимит итераций (7). Попробуйте упростить вопрос.", "", ""

        except Exception as e:
            logger.error(f"ERROR: {str(e)}", exc_info=True)
            return f"❌ Ошибка: {str(e)}", "", ""

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
🧠 Модель: Qwen3-30B-A3B
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
        """Создание Gradio интерфейса"""
        with gr.Blocks(css=MODERN_CSS, title="SMART RAG Agent 2025", theme=gr.themes.Soft()) as interface:

            gr.Markdown("""
            # 🧠 SMART RAG Agent 2025
            ### Qwen3 с Function Calling - Умный многоуровневый поиск
            """)

            # Статус инициализации
            with gr.Row():
                init_status = gr.Textbox(
                    label="🚀 Статус системы",
                    lines=8,
                    interactive=False,
                    value="⏳ Нажмите 'Запустить SMART Agent' для автозагрузки Ultimate базы"
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
                        placeholder="Спросите что угодно - Qwen3 сам решит как искать...",
                        lines=3
                    )
                    ask_btn = gr.Button("✨ Спросить", variant="primary", size="lg")

                with gr.Column(scale=3):
                    answer_output = gr.Textbox(
                        label="🤖 Ответ SMART Agent",
                        lines=18,
                        interactive=False
                    )

            with gr.Row():
                tools_output = gr.Textbox(label="🔧 Использованные инструменты", lines=4, interactive=False)
                memory_info = gr.Textbox(label="📊 Память", lines=4, interactive=False)

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

            gr.Markdown("""
            ---
            ### 💡 Как это работает

            **Qwen3 автоматически:**
            - 🧠 Анализирует ваш вопрос
            - 🔍 Выбирает нужные инструменты (GREP/RAG)
            - 🔄 Делает несколько итераций поиска если нужно
            - ✨ Синтезирует финальный ответ

            **В память сохраняется только финальный ответ (не размышления)**

            **Доступно 3 инструмента:**
            1. `grep_search` - точный поиск с fuzzy
            2. `rag_semantic_search` - семантический векторный поиск
            3. `expand_query` - генерация вариантов написания
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
    print("🧠 SMART RAG Agent 2025 - Qwen3 Function Calling")
    print("="*70)
    print("✨ Умный многоуровневый поиск с автоматическим выбором инструментов")
    print("🗄️ Автозагрузка Ultimate базы (intfloat/multilingual-e5-large)")
    print("💾 В память сохраняются только финальные ответы")
    print("="*70)

    interface.launch(server_name="0.0.0.0", server_port=7861, share=False, show_error=True)


if __name__ == "__main__":
    main()
