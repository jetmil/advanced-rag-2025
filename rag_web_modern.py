"""
Modern RAG Interface 2025 - Glassmorphism + Particles
Оптимизировано для производительности
"""

import gradio as gr
from rag_advanced_memory import AdvancedRAGMemory
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_debug.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Кастомный CSS в стиле 2025
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
    position: relative;
}

/* Анимированные частицы (легковесные) */
body::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image:
        radial-gradient(circle at 20% 50%, rgba(255, 255, 255, 0.05) 0%, transparent 50%),
        radial-gradient(circle at 80% 80%, rgba(255, 255, 255, 0.05) 0%, transparent 50%),
        radial-gradient(circle at 40% 20%, rgba(255, 255, 255, 0.03) 0%, transparent 50%);
    animation: drift 20s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
}

@keyframes drift {
    0%, 100% { transform: translate(0, 0); }
    33% { transform: translate(30px, -30px); }
    66% { transform: translate(-20px, 20px); }
}

/* Основной контейнер - Glassmorphism */
.gradio-container {
    backdrop-filter: blur(10px) !important;
    background: rgba(255, 255, 255, 0.1) !important;
    border-radius: 30px !important;
    box-shadow:
        0 8px 32px 0 rgba(31, 38, 135, 0.37),
        inset 0 1px 1px 0 rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    margin: 20px auto !important;
    padding: 30px !important;
    max-width: 95vw !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

/* Табы - Glassmorphism */
.tab-nav {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 20px !important;
    padding: 5px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

button.selected {
    background: rgba(255, 255, 255, 0.2) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 15px !important;
    color: white !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

/* Карточки */
.gr-box, .gr-form, .gr-panel {
    background: rgba(255, 255, 255, 0.08) !important;
    backdrop-filter: blur(20px) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    box-shadow:
        0 8px 32px 0 rgba(31, 38, 135, 0.2),
        inset 0 1px 1px 0 rgba(255, 255, 255, 0.05) !important;
    padding: 20px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.gr-box:hover {
    background: rgba(255, 255, 255, 0.12) !important;
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(31, 38, 135, 0.3) !important;
}

/* Инпуты и текстовые поля */
.gr-input, .gr-textbox, textarea, input {
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 15px !important;
    color: white !important;
    padding: 12px 16px !important;
    font-size: 14px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    width: 100% !important;
    box-sizing: border-box !important;
    overflow-y: auto !important;
    max-height: 600px !important;
}

.gr-input:focus, .gr-textbox:focus, textarea:focus, input:focus {
    background: rgba(255, 255, 255, 0.15) !important;
    border: 1px solid rgba(255, 255, 255, 0.4) !important;
    box-shadow: 0 0 20px rgba(255, 255, 255, 0.2) !important;
    transform: scale(1.01);
}

input::placeholder, textarea::placeholder {
    color: rgba(255, 255, 255, 0.5) !important;
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
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
}

.gr-button:hover {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.3), rgba(255, 255, 255, 0.2)) !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3) !important;
}

.gr-button:active {
    transform: translateY(0);
}

/* Primary кнопка */
.gr-button-primary {
    background: linear-gradient(135deg, #8B0000 0%, #8B008B 100%) !important;
    border: none !important;
    box-shadow: 0 4px 20px rgba(139, 0, 139, 0.4) !important;
}

.gr-button-primary:hover {
    box-shadow: 0 6px 30px rgba(139, 0, 139, 0.6) !important;
}

/* Слайдеры */
.gr-slider input[type="range"] {
    background: rgba(255, 255, 255, 0.2) !important;
    border-radius: 10px !important;
}

.gr-slider input[type="range"]::-webkit-slider-thumb {
    background: linear-gradient(135deg, #8B0000, #8B008B) !important;
    border-radius: 50% !important;
    box-shadow: 0 2px 10px rgba(139, 0, 139, 0.5) !important;
}

/* Аккордеоны */
.gr-accordion {
    background: rgba(255, 255, 255, 0.05) !important;
    border-radius: 15px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

/* Markdown текст */
.markdown-text, .gr-markdown {
    color: rgba(255, 255, 255, 0.95) !important;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3) !important;
}

.markdown-text h1, .markdown-text h2, .markdown-text h3 {
    color: white !important;
    font-weight: 700 !important;
    text-shadow: 0 2px 20px rgba(0, 0, 0, 0.4) !important;
}

.markdown-text h1 {
    font-size: 2.5em !important;
    background: linear-gradient(135deg, #fff, rgba(255, 255, 255, 0.7));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Лейблы */
label, .gr-label {
    color: rgba(255, 255, 255, 0.9) !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    text-shadow: 0 2px 5px rgba(0, 0, 0, 0.2) !important;
}

/* Прогресс бар */
.progress-bar {
    background: rgba(255, 255, 255, 0.2) !important;
    backdrop-filter: blur(5px) !important;
    border-radius: 10px !important;
}

.progress-bar-fill {
    background: linear-gradient(90deg, #8B0000, #8B008B) !important;
    box-shadow: 0 0 20px rgba(139, 0, 139, 0.6) !important;
}

/* Скроллбар */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 10px;
    backdrop-filter: blur(10px);
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.3);
}

/* Анимация появления */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.gr-box, .gr-button, .gr-input {
    animation: fadeInUp 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Эмодзи эффект */
.emoji {
    display: inline-block;
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}

/* Адаптивность и прокрутка */
.gr-row {
    width: 100% !important;
    max-width: 100% !important;
    overflow-x: hidden !important;
}

.gr-column {
    width: 100% !important;
    max-width: 100% !important;
    overflow-x: hidden !important;
}

/* Прокрутка для длинных текстов */
.gr-textbox textarea {
    overflow-y: auto !important;
    max-height: 400px !important;
}

/* Responsive */
@media (max-width: 1400px) {
    .gradio-container {
        max-width: 98vw !important;
        margin: 10px auto !important;
        padding: 20px !important;
    }
}

@media (max-width: 768px) {
    .gradio-container {
        margin: 5px auto !important;
        padding: 15px !important;
        border-radius: 20px !important;
        max-width: 99vw !important;
    }

    .gr-button {
        padding: 10px 16px !important;
        font-size: 13px !important;
    }

    .gr-input, .gr-textbox, textarea, input {
        font-size: 13px !important;
        padding: 10px 12px !important;
    }
}
"""

class ModernRAGInterface:
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.DEFAULT_DB_PATH = self.project_dir / "chroma_db_kosmoenergy"
        self.DEFAULT_TEXT_FILE = str(self.project_dir / "cosmic_texts.txt")
        self.EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.rag = None
        self.is_initialized = False
        self.current_db_name = "Космоэнергетика"

    def get_available_databases(self):
        """Получение списка доступных баз данных"""
        db_dirs = list(self.project_dir.glob("chroma_db_*"))
        if not db_dirs:
            return []

        db_list = []
        for db_dir in db_dirs:
            # Извлекаем имя базы из папки
            db_name = db_dir.name.replace("chroma_db_", "").replace("_", " ").title()
            db_list.append(db_name)

        return sorted(db_list)

    def detect_embedding_model(self, db_name):
        """Определение embedding модели по имени базы данных"""
        db_name_lower = db_name.lower()

        # Ultimate база использует multilingual-e5-large (1024 dims)
        if "ultimate" in db_name_lower:
            return "intfloat/multilingual-e5-large"

        # Labse база
        if "labse" in db_name_lower:
            return "sentence-transformers/LaBSE"

        # По умолчанию старая модель (384 dims)
        return "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def load_existing_database(self, db_choice, max_short_memory, max_context_tokens, progress=gr.Progress()):
        """Загрузка существующей базы данных"""
        logger.info(f"{'='*70}")
        logger.info(f"ЗАГРУЗКА СУЩЕСТВУЮЩЕЙ БД")
        logger.info(f"Выбрано: {db_choice}")

        if not db_choice:
            logger.error("База не выбрана!")
            return "❌ Выберите базу данных!"

        try:
            progress(0, desc="🔄 Освобождение ресурсов...")
            logger.info("Освобождение ресурсов...")
            self._release_resources()

            # Преобразуем имя обратно в имя папки
            db_name = db_choice.lower().replace(" ", "_")
            db_path = self.project_dir / f"chroma_db_{db_name}"
            logger.info(f"Путь к БД: {db_path}")
            logger.info(f"БД существует: {db_path.exists()}")

            if not db_path.exists():
                logger.error(f"База не найдена: {db_path}")
                return f"❌ База '{db_choice}' не найдена!"

            progress(0.1, desc=f"📚 Загрузка базы '{db_choice}'...")
            logger.info(f"Загрузка базы '{db_choice}'...")

            # Автоопределение embedding модели по имени базы
            embedding_model = self.detect_embedding_model(db_choice)
            logger.info(f"Определена embedding модель: {embedding_model}")

            self.rag = AdvancedRAGMemory(
                text_file_path=self.DEFAULT_TEXT_FILE,
                db_path=str(db_path),
                embedding_model=embedding_model,
                max_short_memory=max_short_memory,
                max_context_tokens=max_context_tokens,
                summarize_threshold=int(max_context_tokens * 0.7),
                enable_auto_summarize=True,
                use_gpu=True
            )

            progress(0.3, desc="🧠 Загрузка embedding модели...")

            from langchain_community.vectorstores import Chroma
            self.rag.vectorstore = Chroma(
                persist_directory=str(db_path),
                embedding_function=self.rag.embeddings
            )

            progress(0.6, desc="🔗 Подключение LM Studio...")
            logger.info("Подключение к LM Studio...")
            self.rag.setup_lm_studio_llm(model_name="google/gemma-3-27b")

            progress(0.8, desc="⚙️ Настройка fuzzy search...")
            logger.info("Настройка retriever с MMR (k=10)...")
            self.rag.create_qa_chain(retriever_k=10, use_mmr=True)

            # Проверка настроек retriever
            logger.info(f"Retriever search_type: {self.rag.retriever.search_type if hasattr(self.rag.retriever, 'search_type') else 'unknown'}")
            logger.info(f"Retriever search_kwargs: {self.rag.retriever.search_kwargs}")

            self.is_initialized = True
            self.current_db_name = db_choice
            progress(1.0, desc="🎉 Готово!")

            logger.info("✅ Загрузка БД завершена успешно!")
            logger.info(f"{'='*70}")

            return f"""✅ База '{db_choice}' загружена успешно!

📁 Путь: {db_path}
💾 Память: {max_short_memory} недавних + автосуммаризация
🎯 Контекст: {max_context_tokens} токенов"""

        except Exception as e:
            logger.error(f"❌ ОШИБКА загрузки БД: {str(e)}", exc_info=True)
            return f"❌ Ошибка загрузки: {str(e)}"

    def _release_resources(self):
        """Освобождение ресурсов и процессов перед инициализацией"""
        try:
            # Освобождаем память старого RAG объекта
            if self.rag is not None:
                del self.rag
                self.rag = None

            # Принудительная сборка мусора Python
            import gc
            gc.collect()

            # Даем время на освобождение ресурсов
            time.sleep(1)

            return True
        except Exception as e:
            print(f"⚠️ Предупреждение при освобождении ресурсов: {e}")
            return False

    def initialize_rag(self, text_file_path, db_name, max_short_memory, max_context_tokens, progress=gr.Progress()):
        logger.info(f"{'='*70}")
        logger.info(f"ИНИЦИАЛИЗАЦИЯ RAG СИСТЕМЫ")
        logger.info(f"Файл: {text_file_path}")
        logger.info(f"Имя БД: {db_name}")
        logger.info(f"Память: short={max_short_memory}, context={max_context_tokens}")

        if not text_file_path or not os.path.exists(text_file_path):
            logger.error(f"Файл не найден: {text_file_path}")
            return "❌ Файл не найден!"

        try:
            # Освобождаем ресурсы перед инициализацией
            progress(0, desc="🔄 Освобождение ресурсов...")
            logger.info("Освобождение ресурсов...")
            self._release_resources()

            project_dir = Path(__file__).parent
            db_path = project_dir / f"chroma_db_{db_name.lower().replace(' ', '_')}"
            logger.info(f"Путь к БД: {db_path}")

            # Проверка существования БД ДО инициализации
            db_exists = os.path.exists(str(db_path))
            logger.info(f"БД существует: {db_exists}")

            if db_exists:
                progress(0.05, desc="✨ Найдена существующая база данных...")
                logger.info("Загрузка существующей базы данных...")
            else:
                progress(0.05, desc="✨ Инициализация новой базы данных...")
                logger.info("Создание новой базы данных...")

            # Автоопределение embedding модели по имени базы
            embedding_model = self.detect_embedding_model(db_name)
            logger.info(f"Определена embedding модель: {embedding_model}")

            self.rag = AdvancedRAGMemory(
                text_file_path=text_file_path,
                db_path=str(db_path),
                embedding_model=embedding_model,
                max_short_memory=max_short_memory,
                max_context_tokens=max_context_tokens,
                summarize_threshold=int(max_context_tokens * 0.7),
                enable_auto_summarize=True,
                use_gpu=True
            )

            progress(0.2, desc="🧠 Загрузка embedding модели...")

            if db_exists:
                progress(0.4, desc="📚 Загрузка существующей базы...")
                from langchain_community.vectorstores import Chroma
                self.rag.vectorstore = Chroma(persist_directory=str(db_path), embedding_function=self.rag.embeddings)
                status_msg = f"✅ База '{db_name}' загружена из кэша (мгновенно!)"
            else:
                progress(0.4, desc="📖 Чтение файла...")
                # Уменьшаем размер чанков и увеличиваем overlap для лучшего поиска
                documents = self.rag.load_and_split_documents(chunk_size=500, chunk_overlap=100)
                progress(0.6, desc="⚡ Векторизация на GPU...")
                self.rag.create_vectorstore(documents, force_recreate=False)
                status_msg = f"✅ База '{db_name}' создана ({len(documents)} чанков)"

            progress(0.8, desc="🔗 Подключение LM Studio...")
            logger.info("Подключение к LM Studio...")
            self.rag.setup_lm_studio_llm(model_name="google/gemma-3-27b")

            progress(0.9, desc="⚙️ Настройка fuzzy search...")
            logger.info("Настройка retriever с MMR (k=10)...")
            # MMR для лучшего покрытия даже с опечатками
            self.rag.create_qa_chain(retriever_k=10, use_mmr=True)

            # Проверка настроек retriever
            logger.info(f"Retriever search_type: {self.rag.retriever.search_type if hasattr(self.rag.retriever, 'search_type') else 'unknown'}")
            logger.info(f"Retriever search_kwargs: {self.rag.retriever.search_kwargs}")

            self.is_initialized = True
            self.current_db_name = db_name
            progress(1.0, desc="🎉 Готово!")

            logger.info("✅ Инициализация завершена успешно!")
            logger.info(f"{'='*70}")

            return f"""✨ Система готова к работе!

{status_msg}
📁 Путь: {db_path}

💾 Память: {max_short_memory} недавних + автосуммаризация
🎯 Контекст: {max_context_tokens} токенов

⚠️ LM Studio должен быть запущен с Gemma-3-27B!"""

        except Exception as e:
            logger.error(f"❌ ОШИБКА инициализации: {str(e)}", exc_info=True)
            return f"❌ Ошибка: {str(e)}"

    def grep_search(self, query: str, context_lines: int = 3):
        """Точный текстовый поиск (аналог grep)"""
        import re

        results = []
        try:
            # Используем файл из RAG (тот который загружен)
            text_file = self.rag.text_file_path if self.rag else self.DEFAULT_TEXT_FILE
            logger.info(f"GREP ищет в файле: {text_file}")

            with open(text_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            pattern = re.compile(query, re.IGNORECASE)

            for i, line in enumerate(lines):
                if pattern.search(line):
                    # Берем контекст
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)

                    context = ''.join(lines[start:end])
                    results.append({
                        'line_num': i + 1,
                        'context': context,
                        'matched_line': line.strip()
                    })

            return results
        except Exception as e:
            logger.error(f"GREP ошибка: {e}")
            return []

    def ask_question(self, question, temperature, max_tokens, num_sources, search_mode):
        logger.info(f"="*70)
        logger.info(f"НОВЫЙ ЗАПРОС: '{question}'")
        logger.info(f"Параметры: temp={temperature}, max_tokens={max_tokens}, num_sources={num_sources}, mode={search_mode}")

        if not self.is_initialized:
            logger.error("Система не инициализирована!")
            return "❌ Сначала инициализируйте систему!", "", "", ""
        if not question.strip():
            logger.error("Пустой вопрос!")
            return "❌ Введите вопрос!", "", "", ""

        try:
            # РЕЖИМ GREP: точный текстовый поиск
            if search_mode == "🔍 GREP":
                logger.info("Режим GREP: точный текстовый поиск")
                grep_results = self.grep_search(question, context_lines=5)

                if not grep_results:
                    return "❌ Ничего не найдено (GREP)", "", "", ""

                # Формируем ответ
                answer = f"🔍 GREP нашел {len(grep_results)} совпадений:\n\n"
                sources = ""

                for i, result in enumerate(grep_results[:20], 1):  # Первые 20
                    answer += f"[{i}] Строка {result['line_num']}: {result['matched_line']}\n\n"
                    sources += f"📄 Совпадение {i} (строка {result['line_num']})\n{result['context']}\n{'='*70}\n\n"

                if len(grep_results) > 20:
                    answer += f"\n... и еще {len(grep_results) - 20} совпадений"

                memory_info = f"🔍 GREP: {len(grep_results)} найдено | Режим: точный поиск"
                context = "\n\n".join([r['context'] for r in grep_results[:20]])

                logger.info(f"GREP: найдено {len(grep_results)} совпадений")
                return answer, sources, memory_info, context

            # РЕЖИМ HYBRID: GREP + RAG
            elif search_mode == "⚡ HYBRID":
                logger.info("Режим HYBRID: GREP + RAG анализ")

                # 1. Сначала GREP для точных совпадений
                grep_results = self.grep_search(question, context_lines=5)

                if not grep_results:
                    return "❌ GREP не нашел совпадений. Попробуйте режим RAG для семантического поиска.", "", "", ""

                # 2. Берем контекст из GREP результатов (ограничиваем для избежания переполнения)
                # Берем максимум 10 результатов или num_sources, что меньше
                max_grep_results = min(10, num_sources)
                grep_contexts = [r['context'] for r in grep_results[:max_grep_results]]
                combined_context = "\n\n".join(grep_contexts)

                # Ограничиваем длину контекста (макс 3000 токенов ~= 12000 символов)
                if len(combined_context) > 12000:
                    combined_context = combined_context[:12000] + "\n\n... (контекст сокращен)"

                logger.info(f"GREP нашел {len(grep_results)} совпадений, отправляем {len(grep_contexts)} в RAG (символов: {len(combined_context)})")

                # 3. RAG анализирует найденные GREP совпадения
                prompt = f"""Ты - эксперт по космоэнергетике.

Вопрос пользователя: {question}

Найденные точные совпадения в тексте:
{combined_context}

Проанализируй эти фрагменты и дай подробный ответ на вопрос пользователя."""

                response = self.rag.llm_client.chat.completions.create(
                    model=self.rag.model_name,
                    messages=[
                        {"role": "system", "content": "Ты - эксперт по космоэнергетике, анализируешь точные совпадения из текста."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=int(max_tokens),
                    temperature=temperature
                )

                answer = f"⚡ HYBRID: найдено {len(grep_results)} точных совпадений (GREP), анализ RAG:\n\n" + response.choices[0].message.content

                sources = f"🔍 GREP нашел {len(grep_results)} совпадений:\n\n"
                for i, result in enumerate(grep_results[:10], 1):
                    sources += f"📄 Совпадение {i} (строка {result['line_num']})\n{result['context'][:400]}...\n\n"

                memory_info = f"⚡ HYBRID: GREP {len(grep_results)} → RAG анализ"

                logger.info("HYBRID успешно: GREP + RAG анализ")
                return answer, sources, memory_info, combined_context

            # РЕЖИМ RAG: семантический поиск + LLM (по умолчанию)
            else:  # search_mode == "🤖 RAG"
                logger.info("Режим RAG: семантический поиск + LLM")

                # Обновляем search_kwargs с учетом MMR
                logger.info(f"Текущая база данных: {self.current_db_name}")
                logger.info(f"Путь к БД: {self.rag.db_path}")

                search_kwargs = {
                    "k": num_sources,
                    "fetch_k": num_sources * 3,  # Больше кандидатов для fuzzy search
                    "lambda_mult": 0.5
                }
                logger.info(f"search_kwargs: {search_kwargs}")
                self.rag.retriever.search_kwargs = search_kwargs

                logger.info("Начало поиска релевантных документов...")
                # Сначала ищем документы напрямую для проверки
                test_docs = self.rag.retriever.get_relevant_documents(question)
                logger.info(f"Найдено документов: {len(test_docs)}")

                for i, doc in enumerate(test_docs[:3], 1):
                    preview = doc.page_content[:200].replace('\n', ' ')
                    logger.debug(f"Документ {i}: {preview}...")

                logger.info("Отправка запроса к LLM...")
                result = self.rag.query(question, max_tokens=int(max_tokens), temperature=temperature)

                logger.info(f"Получен ответ от LLM (длина: {len(result['answer'])} символов)")
                logger.debug(f"Ответ: {result['answer'][:200]}...")

                sources = ""
                for i, doc in enumerate(result['source_documents'], 1):
                    content = doc.page_content[:400]
                    sources += f"📄 Источник {i}\n{content}{'...' if len(doc.page_content) > 400 else ''}\n\n"

                stats = result['memory_stats']
                memory_info = f"""💾 Память: {stats['short_memory_size']} недавних | {stats['long_memory_size']} суммаризированных
📊 Токены: {stats['tokens_used']}/{stats['tokens_limit']} ({int(stats['tokens_used']/stats['tokens_limit']*100)}%)"""

                logger.info("Запрос успешно обработан")
                logger.info(f"="*70)
                return result['answer'], sources, memory_info, result.get('context', '')

        except Exception as e:
            logger.error(f"ОШИБКА при обработке запроса: {str(e)}", exc_info=True)
            error = f"❌ Ошибка: {str(e)}"
            if "connection" in str(e).lower():
                error += "\n\n⚠️ Проверьте LM Studio!"
            return error, "", "", ""

    def clear_memory(self, keep_summaries):
        if not self.is_initialized:
            return "❌ Система не инициализирована!"
        return f"✅ {self.rag.clear_memory(keep_summaries=keep_summaries)}"

    def get_stats(self):
        if not self.is_initialized:
            return "❌ Система не инициализирована!"

        stats = self.rag.get_memory_stats()
        return f"""📊 Статистика сессии

🕐 Длительность: {stats['session_duration']}
💬 Всего вопросов: {stats['total_questions']}
📝 Короткая память: {stats['short_memory_count']}
📚 Долгая память: {stats['long_memory_count']}

💾 База: {self.current_db_name}
⚙️ Автосуммаризация: {'✅' if stats['auto_summarize_enabled'] else '❌'}"""

    def export_history(self):
        if not self.is_initialized:
            return "❌ Система не инициализирована!", None

        filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        project_dir = Path(__file__).parent
        filepath = project_dir / filename
        result = self.rag.export_conversation(str(filepath))
        return f"✅ {result}", str(filepath)

    def create_interface(self):
        with gr.Blocks(css=MODERN_CSS, title="Modern RAG 2025", theme=gr.themes.Soft()) as interface:

            gr.Markdown("""
            # <span class="emoji">🔮</span> Advanced RAG Knowledge Base
            ### Glassmorphism Edition 2025
            """)

            with gr.Tab("🚀 Инициализация"):
                gr.Markdown("### Настройка вашей базы знаний")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 📥 Загрузить существующую базу")
                        db_dropdown = gr.Dropdown(
                            label="📚 Выберите базу данных",
                            choices=self.get_available_databases(),
                            value=None,
                            interactive=True
                        )
                        refresh_db_btn = gr.Button("🔄 Обновить список", size="sm")
                        load_db_btn = gr.Button("📥 Загрузить базу", variant="secondary", size="lg")

                        gr.Markdown("---")
                        gr.Markdown("#### ✨ Или создать новую базу")

                        text_file_input = gr.Textbox(
                            label="📁 Путь к файлу",
                            value=self.DEFAULT_TEXT_FILE,
                            placeholder="C:\\путь\\к\\файлу.txt"
                        )
                        db_name_input = gr.Textbox(
                            label="📚 Название новой базы",
                            value="Космоэнергетика"
                        )

                        with gr.Accordion("⚙️ Настройки памяти", open=False):
                            max_short_memory = gr.Slider(3, 10, 5, 1, label="Короткая память")
                            max_context = gr.Slider(6000, 16000, 8000, 1000, label="Макс токенов контекста")

                        init_btn = gr.Button("✨ Инициализировать новую", variant="primary", size="lg")

                    with gr.Column():
                        init_status = gr.Textbox(label="Статус", lines=12, interactive=False)

            with gr.Tab("💬 Чат"):
                with gr.Row():
                    with gr.Column(scale=2):
                        question_input = gr.Textbox(
                            label="💭 Ваш вопрос",
                            placeholder="Спросите что-нибудь...",
                            lines=3
                        )

                        search_mode = gr.Radio(
                            choices=["⚡ HYBRID", "🤖 RAG", "🔍 GREP"],
                            value="⚡ HYBRID",
                            label="🔎 Режим поиска",
                            info="HYBRID=умный поиск+анализ | RAG=семантика | GREP=точный текст"
                        )

                        with gr.Accordion("⚙️ Параметры", open=False):
                            temperature = gr.Slider(0, 1, 0.7, 0.1, label="Temperature")
                            max_tokens = gr.Slider(500, 4000, 2000, 100, label="Max tokens")
                            num_sources = gr.Slider(1, 100, 20, 5, label="Источников (больше = полнее)")

                        ask_btn = gr.Button("✨ Спросить", variant="primary", size="lg")

                    with gr.Column(scale=3):
                        answer_output = gr.Textbox(label="💬 Ответ", lines=15, interactive=False)

                with gr.Row():
                    memory_info = gr.Textbox(label="📊 Память", lines=2, interactive=False)

                with gr.Accordion("📚 Источники", open=False):
                    sources_output = gr.Textbox(lines=8, interactive=False)

                with gr.Accordion("🔍 Контекст", open=False):
                    context_output = gr.Textbox(lines=8, interactive=False)

            with gr.Tab("📊 Память"):
                gr.Markdown("### Управление памятью диалога")

                with gr.Row():
                    stats_btn = gr.Button("📊 Статистика", size="lg")
                    clear_btn = gr.Button("🧹 Очистить (сохранить суммарии)", size="lg")
                    clear_all_btn = gr.Button("🗑️ Очистить всё", variant="stop", size="lg")

                stats_output = gr.Textbox(label="Статистика", lines=10, interactive=False)

                with gr.Row():
                    export_btn = gr.Button("💾 Экспорт истории", size="lg")

                export_status = gr.Textbox(label="Статус экспорта", lines=2)
                export_file = gr.File(label="Скачать")

            gr.Markdown("""
            ---
            ### 💡 Возможности
            - ✨ Умная память с автосуммаризацией
            - ⚡ GPU-ускорение (RTX 3090)
            - 🎨 Glassmorphism UI 2025
            - 💾 Экспорт истории диалога
            """)

            # Events
            # Инициализация
            init_btn.click(self.initialize_rag, [text_file_input, db_name_input, max_short_memory, max_context], [init_status])
            load_db_btn.click(self.load_existing_database, [db_dropdown, max_short_memory, max_context], [init_status])
            refresh_db_btn.click(lambda: gr.Dropdown(choices=self.get_available_databases()), outputs=[db_dropdown])

            # Чат
            ask_btn.click(self.ask_question, [question_input, temperature, max_tokens, num_sources, search_mode], [answer_output, sources_output, memory_info, context_output])
            question_input.submit(self.ask_question, [question_input, temperature, max_tokens, num_sources, search_mode], [answer_output, sources_output, memory_info, context_output])

            # Память
            stats_btn.click(self.get_stats, outputs=[stats_output])
            clear_btn.click(lambda: self.clear_memory(True), outputs=[stats_output])
            clear_all_btn.click(lambda: self.clear_memory(False), outputs=[stats_output])
            export_btn.click(self.export_history, outputs=[export_status, export_file])

        return interface


def main():
    app = ModernRAGInterface()
    interface = app.create_interface()

    print("="*70)
    print("🚀 Modern RAG 2025 - Glassmorphism Edition")
    print("="*70)

    interface.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)


if __name__ == "__main__":
    main()
