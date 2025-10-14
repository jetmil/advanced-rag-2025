"""
Продвинутый веб-интерфейс RAG с памятью и мультибазами
"""

import gradio as gr
from rag_advanced_memory import AdvancedRAGMemory
import os
from pathlib import Path
from datetime import datetime

class AdvancedRAGInterface:
    def __init__(self):
        # Настройки по умолчанию
        project_dir = Path(__file__).parent
        self.DEFAULT_DB_PATH = project_dir / "chroma_db_kosmoenergy"
        self.DEFAULT_TEXT_FILE = r"C:\Users\PC\Downloads\consolidated_texts_20251014_235421_cleaned.txt"
        self.EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

        self.rag = None
        self.is_initialized = False
        self.current_db_name = "Космоэнергетика"

    def initialize_rag(
        self,
        text_file_path: str,
        db_name: str,
        max_short_memory: int,
        max_context_tokens: int,
        progress=gr.Progress()
    ):
        """Инициализация RAG системы"""

        if not text_file_path or not os.path.exists(text_file_path):
            return "❌ Файл не найден! Укажите правильный путь."

        try:
            project_dir = Path(__file__).parent
            db_path = project_dir / f"chroma_db_{db_name.lower().replace(' ', '_')}"

            progress(0, desc="Инициализация...")

            # Создание RAG системы
            self.rag = AdvancedRAGMemory(
                text_file_path=text_file_path,
                db_path=str(db_path),
                embedding_model=self.EMBEDDING_MODEL,
                max_short_memory=max_short_memory,
                max_context_tokens=max_context_tokens,
                summarize_threshold=int(max_context_tokens * 0.7),
                enable_auto_summarize=True,
                use_gpu=True
            )

            progress(0.2, desc="Загрузка embedding модели...")

            # Проверка существования БД
            if os.path.exists(str(db_path)):
                progress(0.4, desc="Загрузка существующей БД...")
                from langchain_community.vectorstores import Chroma
                self.rag.vectorstore = Chroma(
                    persist_directory=str(db_path),
                    embedding_function=self.rag.embeddings
                )
                status_msg = f"✅ База данных '{db_name}' загружена\n📁 Путь: {db_path}"
            else:
                progress(0.4, desc="Чтение файла...")
                documents = self.rag.load_and_split_documents(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                progress(0.6, desc="Создание векторов на GPU...")
                self.rag.create_vectorstore(documents, force_recreate=False)
                status_msg = f"✅ База данных '{db_name}' создана\n📁 Путь: {db_path}\n📊 Документов: {len(documents)}"

            progress(0.8, desc="Подключение к LM Studio...")
            self.rag.setup_lm_studio_llm(model_name="google/gemma-3-27b")

            progress(0.9, desc="Настройка retriever...")
            self.rag.create_qa_chain(retriever_k=4)

            self.is_initialized = True
            self.current_db_name = db_name

            progress(1.0, desc="Готово!")

            return f"""✅ RAG система инициализирована!

{status_msg}

⚙️ Настройки памяти:
- Короткая память: {max_short_memory} сообщений
- Макс контекст: {max_context_tokens} токенов
- Автосуммаризация: Включена

⚠️ Убедитесь, что LM Studio запущен с Gemma-3-27B!"""

        except Exception as e:
            return f"❌ Ошибка инициализации:\n{str(e)}"

    def ask_question(self, question, temperature, max_tokens, num_sources):
        """Обработка вопроса"""
        if not self.is_initialized:
            return "❌ Сначала инициализируйте систему!", "", "", ""

        if not question.strip():
            return "❌ Введите вопрос!", "", "", ""

        try:
            # Обновление параметров
            self.rag.retriever.search_kwargs = {"k": num_sources}

            # Запрос
            result = self.rag.query(
                question=question,
                max_tokens=int(max_tokens),
                temperature=temperature
            )

            # Форматирование ответа
            answer = result['answer']

            # Источники
            sources = ""
            for i, doc in enumerate(result['source_documents'], 1):
                sources += f"\n--- Источник {i} ---\n"
                sources += doc.page_content[:400]
                if len(doc.page_content) > 400:
                    sources += "...\n"
                else:
                    sources += "\n"

            # Статистика памяти
            stats = result['memory_stats']
            memory_info = f"""📊 Статистика памяти:
- Недавних сообщений: {stats['short_memory_size']}
- Суммаризированных блоков: {stats['long_memory_size']}
- Использовано токенов: {stats['tokens_used']}/{stats['tokens_limit']}
- Загрузка: {int(stats['tokens_used']/stats['tokens_limit']*100)}%"""

            # Контекст
            context = result.get('context', '')

            return answer, sources, memory_info, context

        except Exception as e:
            error_msg = f"❌ Ошибка: {str(e)}"
            if "Connection" in str(e) or "connect" in str(e).lower():
                error_msg += "\n\n⚠️ Проверьте, что LM Studio запущен!"
            return error_msg, "", "", ""

    def clear_memory(self, keep_summaries: bool):
        """Очистка памяти"""
        if not self.is_initialized:
            return "❌ Система не инициализирована!"

        result = self.rag.clear_memory(keep_summaries=keep_summaries)
        return f"✅ {result}"

    def get_stats(self):
        """Получение статистики"""
        if not self.is_initialized:
            return "❌ Система не инициализирована!"

        stats = self.rag.get_memory_stats()
        return f"""📊 Статистика сессии:

🕐 Длительность сессии: {stats['session_duration']}
💬 Всего вопросов: {stats['total_questions']}
📝 В короткой памяти: {stats['short_memory_count']}
📚 В долгой памяти: {stats['long_memory_count']}

⚙️ Настройки:
- Автосуммаризация: {'Включена' if stats['auto_summarize_enabled'] else 'Выключена'}
- Макс токенов контекста: {stats['max_context_tokens']}
- Порог суммаризации: {stats['summarize_threshold']}

💾 База данных: {self.current_db_name}
"""

    def export_history(self):
        """Экспорт истории"""
        if not self.is_initialized:
            return "❌ Система не инициализирована!", None

        filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        project_dir = Path(__file__).parent
        filepath = project_dir / filename

        result = self.rag.export_conversation(str(filepath))

        return f"✅ {result}", filepath

    def create_interface(self):
        """Создание Gradio интерфейса"""

        with gr.Blocks(title="Advanced RAG - Космоэнергетика", theme=gr.themes.Soft()) as interface:

            gr.Markdown("""
            # 🔮 Advanced RAG Knowledge Base

            Продвинутая система с памятью диалога и автосуммаризацией.

            ## Возможности:
            - ✅ Умная память с автосуммаризацией
            - ✅ Оптимизация под RTX 3090 (24GB) + 32GB RAM
            - ✅ Поддержка длинных диалогов
            - ✅ Экспорт истории разговора
            - ✅ Мультибазы знаний
            """)

            with gr.Tab("🚀 Инициализация"):
                gr.Markdown("""
                ### Настройка базы знаний

                **Что происходит при инициализации:**
                1. Загрузка embedding модели на GPU (~20 сек)
                2. Чтение вашего текстового файла
                3. Разбивка на чанки и векторизация
                4. Сохранение в ChromaDB (переиспользуется!)
                """)

                with gr.Row():
                    with gr.Column():
                        text_file_input = gr.Textbox(
                            label="📁 Путь к текстовому файлу",
                            value=self.DEFAULT_TEXT_FILE,
                            placeholder="C:\\путь\\к\\файлу.txt"
                        )
                        db_name_input = gr.Textbox(
                            label="📚 Название базы данных",
                            value="Космоэнергетика",
                            placeholder="Моя база знаний"
                        )

                        with gr.Accordion("⚙️ Настройки памяти", open=False):
                            max_short_memory_slider = gr.Slider(
                                minimum=3,
                                maximum=10,
                                value=5,
                                step=1,
                                label="Короткая память (последние N сообщений)"
                            )
                            max_context_tokens_slider = gr.Slider(
                                minimum=4000,
                                maximum=7000,
                                value=6000,
                                step=500,
                                label="Макс токенов контекста (для Gemma-27B: 8192)"
                            )

                        init_btn = gr.Button("🚀 Инициализировать систему", variant="primary", size="lg")

                    with gr.Column():
                        init_status = gr.Textbox(
                            label="Статус инициализации",
                            placeholder="Нажмите кнопку инициализации...",
                            lines=10,
                            interactive=False
                        )

            with gr.Tab("💬 Чат"):
                with gr.Row():
                    with gr.Column(scale=2):
                        question_input = gr.Textbox(
                            label="Ваш вопрос",
                            placeholder="Например: Какие упражнения помогают развить видение ауры?",
                            lines=3
                        )

                        with gr.Accordion("⚙️ Параметры генерации", open=False):
                            temperature = gr.Slider(0.0, 1.0, 0.7, 0.1, label="Temperature")
                            max_tokens = gr.Slider(500, 4000, 2000, 100, label="Max tokens")
                            num_sources = gr.Slider(1, 10, 4, 1, label="Количество источников")

                        ask_btn = gr.Button("💬 Задать вопрос", variant="primary", size="lg")

                    with gr.Column(scale=3):
                        answer_output = gr.Textbox(
                            label="Ответ",
                            lines=15,
                            interactive=False
                        )

                with gr.Row():
                    with gr.Column():
                        memory_info_output = gr.Textbox(
                            label="📊 Статистика памяти",
                            lines=5,
                            interactive=False
                        )

                with gr.Accordion("📚 Источники информации", open=False):
                    sources_output = gr.Textbox(lines=10, interactive=False)

                with gr.Accordion("🔍 Полный контекст", open=False):
                    context_output = gr.Textbox(lines=10, interactive=False)

            with gr.Tab("📊 Управление памятью"):
                gr.Markdown("""
                ### Управление памятью диалога

                **Короткая память:** Последние N сообщений в полном виде
                **Долгая память:** Суммаризированная история старых сообщений
                **Автосуммаризация:** Срабатывает при превышении порога токенов
                """)

                with gr.Row():
                    stats_btn = gr.Button("📊 Показать статистику", size="lg")
                    clear_btn = gr.Button("🧹 Очистить память (сохранить суммарии)", size="lg")
                    clear_all_btn = gr.Button("🗑️ Очистить всю память", variant="stop", size="lg")

                stats_output = gr.Textbox(
                    label="Статистика памяти",
                    lines=12,
                    interactive=False
                )

                gr.Markdown("---")

                with gr.Row():
                    export_btn = gr.Button("💾 Экспорт истории разговора", size="lg")

                with gr.Row():
                    export_status = gr.Textbox(label="Статус экспорта", lines=2)
                    export_file = gr.File(label="Скачать файл")

            gr.Markdown("""
            ---
            ### 💡 Примеры вопросов:
            - Как правильно настроить взгляд для видения ауры?
            - Какие упражнения для глаз рекомендуются?
            - Расскажи про биолокационную рамку
            - Как проводить диагностику энергоцентров?

            ### 🎯 Оптимизация:
            - **GPU:** RTX 3090 (24GB VRAM)
            - **RAM:** 32GB
            - **LLM:** Google Gemma-3-27B
            - **Context:** 8192 tokens
            - **Embedding:** GPU-ускоренный
            """)

            # Обработчики событий
            init_btn.click(
                fn=self.initialize_rag,
                inputs=[text_file_input, db_name_input, max_short_memory_slider, max_context_tokens_slider],
                outputs=[init_status]
            )

            ask_btn.click(
                fn=self.ask_question,
                inputs=[question_input, temperature, max_tokens, num_sources],
                outputs=[answer_output, sources_output, memory_info_output, context_output]
            )

            question_input.submit(
                fn=self.ask_question,
                inputs=[question_input, temperature, max_tokens, num_sources],
                outputs=[answer_output, sources_output, memory_info_output, context_output]
            )

            stats_btn.click(
                fn=self.get_stats,
                outputs=[stats_output]
            )

            clear_btn.click(
                fn=lambda: self.clear_memory(keep_summaries=True),
                outputs=[stats_output]
            )

            clear_all_btn.click(
                fn=lambda: self.clear_memory(keep_summaries=False),
                outputs=[stats_output]
            )

            export_btn.click(
                fn=self.export_history,
                outputs=[export_status, export_file]
            )

        return interface


def main():
    app = AdvancedRAGInterface()
    interface = app.create_interface()

    print("\n" + "="*70)
    print("Запуск Advanced RAG Web Interface...")
    print("="*70 + "\n")

    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
