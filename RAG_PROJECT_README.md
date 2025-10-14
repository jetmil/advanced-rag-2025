# 🔮 Advanced RAG Knowledge Base - Modern Edition 2025

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red)
![CUDA](https://img.shields.io/badge/CUDA-12.4-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

Продвинутая система RAG (Retrieval-Augmented Generation) с умной памятью, автосуммаризацией и современным glassmorphism UI.

## ✨ Особенности

- 🧠 **Умная память диалога** с автоматической суммаризацией
- ⚡ **GPU-ускорение** для embeddings (оптимизировано под RTX 3090)
- 💾 **ChromaDB** для persistent векторного хранилища
- 🎨 **Glassmorphism UI 2025** с частицами
- 📊 **Продвинутая статистика** использования токенов и памяти
- 💬 **Контекстные ответы** с учетом истории разговора
- 📁 **Мультибазы знаний** - поддержка нескольких тем
- 🔄 **Экспорт истории** разговора
- 🌐 **Веб-интерфейс** на Gradio

---

## 🎯 Для кого этот проект?

- Исследователи AI и NLP
- Разработчики RAG-систем
- Специалисты по работе со знаниями
- Энтузиасты локальных LLM
- Те, кто хочет создать свою базу знаний

---

## 🚀 Быстрый старт

### Требования

- **GPU:** NVIDIA RTX 3090 (24GB) или аналог
- **RAM:** 32GB (рекомендуется)
- **OS:** Windows 10/11
- **Python:** 3.13+
- **CUDA:** 12.4+
- **LM Studio:** установлен и настроен

### Установка

```bash
# 1. Клонирование репозитория
git clone https://github.com/jetmil/advanced-rag-2025.git
cd advanced-rag-2025

# 2. Установка зависимостей
pip install -r requirements.txt

# 3. Установка PyTorch с CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Запуск LM Studio сервера
start_lmstudio_server.bat

# 5. Запуск веб-интерфейса
python rag_web_modern.py
```

### Первый запуск

1. Откройте браузер: `http://localhost:7860`
2. Перейдите на вкладку "🚀 Инициализация"
3. Укажите путь к вашему текстовому файлу
4. Нажмите "✨ Инициализировать"
5. Начинайте задавать вопросы!

---

## 📁 Структура проекта

```
advanced-rag-2025/
├── rag_knowledge_base.py          # Базовый RAG класс
├── rag_advanced_memory.py         # RAG с умной памятью
├── rag_web_modern.py              # Modern UI 2025
├── start_lmstudio_server.bat      # Запуск LM Studio CLI
├── check_lmstudio_status.bat      # Проверка статуса
├── stop_lmstudio_server.bat       # Остановка сервера
├── requirements.txt               # Python зависимости
├── README.md                      # Эта документация
├── TROUBLESHOOTING.md             # Решение проблем
└── chroma_db_*/                   # Векторные базы данных
```

---

## 🏗️ Архитектура

```
┌─────────────────────────────────────────────────────────┐
│                    Пользователь                          │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│              Gradio Web Interface                        │
│           (Glassmorphism UI 2025)                        │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│         Advanced RAG Memory System                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │Short Memory  │  │Long Memory   │  │Auto-Summarize│  │
│  │(Last 5 msgs) │  │(Summaries)   │  │(4000 tokens) │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└───────────────────┬─────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌───────────────┐     ┌───────────────────┐
│  Embeddings   │     │   LM Studio API   │
│  (GPU-accel)  │     │  (Gemma-3-27B)    │
│               │     │                   │
│ nomic-embed   │     │ OpenAI-compatible │
│ on RTX 3090   │     │   localhost:1234  │
└───────┬───────┘     └─────────┬─────────┘
        │                       │
        ▼                       │
┌───────────────┐               │
│   ChromaDB    │               │
│  (Persistent) │               │
│               │               │
│ Vector Store  │               │
└───────────────┘               │
        │                       │
        └───────────┬───────────┘
                    │
                    ▼
            ┌───────────────┐
            │   Response    │
            │ + Sources     │
            │ + Statistics  │
            └───────────────┘
```

---

## 💡 Как это работает

### 1. RAG (Retrieval-Augmented Generation)

```
Вопрос пользователя
    ↓
Векторизация вопроса (GPU)
    ↓
Поиск в ChromaDB (top-4 похожих чанка)
    ↓
Формирование промпта (вопрос + контекст + история)
    ↓
Отправка в Gemma-3-27B (LM Studio)
    ↓
Ответ с учетом контекста
```

### 2. Умная память

```
Сообщение 1-5: Короткая память (полный текст)
    ↓
Превышен порог токенов (4000)
    ↓
Автосуммаризация сообщений 1-3
    ↓
Сохранение в долгую память
    ↓
Освобождение места для новых сообщений
```

### 3. Оптимизация токенов

```python
max_context_tokens = 6000  # Из 8192 (Gemma-27B)
├─ Base prompt: ~500 tokens
├─ Context from DB: ~2000 tokens
├─ Short memory: ~1500 tokens
├─ Long memory: ~500 tokens
└─ Reserve: ~1500 tokens
```

---

## ⚙️ Настройки под вашу систему

### RTX 3090 (24GB VRAM) + 32GB RAM

**Оптимальные параметры:**

```python
# В rag_web_modern.py
max_short_memory = 5          # Последние 5 сообщений
max_context_tokens = 6000     # Из 8192 (Gemma context)
summarize_threshold = 4000    # Суммаризация при 4000 токенов
chunk_size = 1000             # Размер чанка
chunk_overlap = 200           # Перекрытие чанков
retriever_k = 4               # Количество источников
```

### Другие GPU

**RTX 4090 (24GB):**
- Такие же настройки

**RTX 3080 (10GB):**
```python
max_short_memory = 3
max_context_tokens = 4000
```

**RTX 3060 (12GB):**
```python
max_short_memory = 4
max_context_tokens = 5000
```

---

## 🔧 Конфигурация

### 1. Выбор embedding модели

```python
# Быстрая (~120MB VRAM)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Качественная (~470MB VRAM)
EMBEDDING_MODEL = "sentence-transformers/LaBSE"

# Топовая (~2.2GB VRAM)
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
```

### 2. Выбор LLM модели

```python
# В LM Studio загрузите одну из:
- google/gemma-3-27b          # 27B параметров, context 8192
- qwen/qwen3-30b-a3b-2507     # 30B параметров, context 32768
- mistral/mistral-large        # 123B параметров, context 32768
```

### 3. Настройка чанкинга

```python
# Для коротких ответов
chunk_size = 500
chunk_overlap = 100

# Для длинных контекстов
chunk_size = 1500
chunk_overlap = 300
```

---

## 📊 Преодоленные проблемы

### 1. ❌ PyTorch без CUDA

**Проблема:**
```
RuntimeError: Torch not compiled with CUDA enabled
```

**Решение:**
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 2. ❌ Ошибка импорта sentence-transformers

**Проблема:**
```
ModuleNotFoundError: No module named 'multiprocess'
```

**Решение:**
```bash
pip install multiprocess dill xxhash
```

### 3. ❌ Конфликт torchvision

**Проблема:**
```
RuntimeError: operator torchvision::nms does not exist
```

**Решение:**
```bash
pip install torchvision --upgrade --force-reinstall
```

### 4. ❌ LM Studio не отвечает

**Проблема:**
```
Connection refused to localhost:1234
```

**Решение:**
1. Проверить, что LM Studio запущен
2. Проверить, что модель загружена
3. Проверить, что сервер запущен (Local Server → Start)
4. Проверить порт (должен быть 1234)

### 5. ❌ Out of Memory на GPU

**Проблема:**
```
CUDA out of memory
```

**Решение:**
1. Закрыть другие приложения на GPU
2. Уменьшить `max_context_tokens`
3. Уменьшить `retriever_k`
4. Использовать меньшую embedding модель

### 6. ❌ Медленная работа

**Причины и решения:**

| Проблема | Решение |
|----------|---------|
| Embedding на CPU | Установить PyTorch с CUDA |
| LLM не на GPU | В LM Studio: GPU Offload = Max |
| Большой контекст | Уменьшить `max_context_tokens` |
| Много источников | Уменьшить `retriever_k` |

---

## 📚 Примеры использования

### 1. Базовый запрос

```python
from rag_advanced_memory import AdvancedRAGMemory

rag = AdvancedRAGMemory(
    text_file_path="my_knowledge.txt",
    db_path="chroma_db",
    use_gpu=True
)

# Инициализация
rag.setup_lm_studio_llm()
rag.create_qa_chain()

# Запрос
result = rag.query("Как работает RAG?")
print(result['answer'])
```

### 2. С кастомными настройками

```python
rag = AdvancedRAGMemory(
    text_file_path="my_knowledge.txt",
    db_path="chroma_db",
    max_short_memory=7,
    max_context_tokens=7000,
    summarize_threshold=5000,
    use_gpu=True
)
```

### 3. Работа с памятью

```python
# Запрос с историей
result = rag.query("Расскажи про embeddings")
result = rag.query("А как они создаются?")  # Помнит контекст!

# Статистика
stats = rag.get_memory_stats()
print(f"Вопросов задано: {stats['total_questions']}")

# Очистка памяти
rag.clear_memory(keep_summaries=True)

# Экспорт
rag.export_conversation("history.txt")
```

---

## 🎨 UI Features

### Glassmorphism Design 2025

- **Прозрачные панели** с размытием фона
- **Градиентные кнопки** с hover-эффектами
- **Анимированные частицы** (легковесные, не тормозят)
- **Плавные переходы** между состояниями
- **Адаптивный дизайн** под разные экраны
- **Темная тема** с яркими акцентами

### Оптимизация производительности

```css
/* Легковесные частицы - только CSS */
body::before {
    animation: drift 20s ease-in-out infinite;
}

/* Hardware acceleration */
transform: translateZ(0);
will-change: transform;

/* Оптимизированные градиенты */
background: linear-gradient(135deg, ...);
```

---

## 🛠️ Дополнительные инструменты

### 1. Управление LM Studio через CLI

```bash
# Запуск сервера
start_lmstudio_server.bat

# Проверка статуса
check_lmstudio_status.bat

# Остановка
stop_lmstudio_server.bat
```

### 2. Консольная версия

```bash
# Запуск RAG без GUI
python rag_advanced_memory.py

# Команды:
# - 'stats' - статистика
# - 'clear' - очистка памяти
# - 'export' - экспорт истории
# - 'quit' - выход
```

---

## 📈 Производительность

### Бенчмарки (RTX 3090 + 32GB RAM)

| Операция | Время | Примечание |
|----------|-------|------------|
| Загрузка embedding модели | ~20 сек | Первый раз |
| Векторизация 1000 чанков | ~1-2 мин | GPU-ускоренная |
| Поиск в ChromaDB | <100 мс | Очень быстро |
| Генерация ответа (Gemma-27B) | 3-10 сек | Зависит от длины |
| Автосуммаризация | ~5 сек | LLM запрос |

### Использование ресурсов

```
GPU (RTX 3090 24GB):
├─ Gemma-27B: ~16GB
├─ Embeddings: ~2GB
└─ Свободно: ~6GB

RAM (32GB):
├─ ChromaDB: ~500MB
├─ Python: ~1GB
├─ Система: ~4GB
└─ Свободно: ~26GB
```

---

## 🔒 Безопасность и приватность

### Полностью локальное решение

- ✅ Все данные хранятся локально
- ✅ Нет отправки данных в облако
- ✅ LM Studio работает оффлайн
- ✅ ChromaDB - локальная база данных
- ✅ Embedding модели загружаются один раз

### Рекомендации

1. **Не коммитьте** `.env` файлы с секретами
2. **Добавьте в .gitignore:**
   ```
   chroma_db_*/
   *.log
   conversation_*.txt
   ```
3. **Бэкапьте** векторные базы данных
4. **Шифруйте** чувствительные данные

---

## 🤝 Вклад в проект

Приветствуются pull requests! Для крупных изменений сначала откройте issue.

### Как внести вклад

1. Fork проекта
2. Создайте feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit изменений (`git commit -m 'Add some AmazingFeature'`)
4. Push в branch (`git push origin feature/AmazingFeature`)
5. Откройте Pull Request

---

## 📜 Лицензия

MIT License - смотрите [LICENSE](LICENSE) файл

---

## 🙏 Благодарности

- **HuggingFace** за sentence-transformers
- **Chroma** за векторную БД
- **LangChain** за RAG фреймворк
- **LM Studio** за удобный интерфейс для LLM
- **Gradio** за быстрый UI

---

## 📞 Контакты

**GitHub:** [@jetmil](https://github.com/jetmil)

**Проект:** https://github.com/jetmil/advanced-rag-2025

---

## 🗺️ Roadmap

- [ ] Поддержка PDF и DOCX
- [ ] Мультиязычность интерфейса
- [ ] API endpoints (FastAPI)
- [ ] Docker контейнер
- [ ] Интеграция с Ollama
- [ ] Поддержка изображений (multimodal)
- [ ] Голосовой ввод
- [ ] Телеграм бот

---

## 📝 Changelog

### v1.0.0 (2025-10-15)

- ✅ Базовый RAG функционал
- ✅ Умная память с автосуммаризацией
- ✅ Glassmorphism UI 2025
- ✅ Оптимизация под RTX 3090
- ✅ Экспорт истории
- ✅ Мультибазы знаний

---

<div align="center">

**Сделано с ❤️ и ☕ в 2025**

[⬆ Вернуться наверх](#-advanced-rag-knowledge-base---modern-edition-2025)

</div>
