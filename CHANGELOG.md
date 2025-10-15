# Changelog - Advanced RAG 2025

## [2025-10-16] - SMART Agent Release 🧠

### Новые возможности

#### 🧠 SMART RAG Agent с Qwen3 Function Calling
- **Умный агент** с многоуровневой логикой поиска
- **Qwen3-30B-A3B** сама выбирает инструменты и стратегию
- **Function Calling** - автоматический вызов grep/RAG/expand_query
- **Итеративное уточнение** - до 7 раундов поиска
- **Автозагрузка** Ultimate базы при старте
- **16000 токенов** контекста (вдвое больше чем было)
- **Только финальные ответы** в память (не размышления)

#### 🔧 Доступные инструменты для Qwen3
1. `grep_search` - точный fuzzy поиск в тексте
2. `rag_semantic_search` - семантический векторный поиск
3. `expand_query` - генерация вариантов написания

#### 📁 Новые файлы
- `rag_smart_qwen.py` - основной файл SMART Agent (680 строк)
- `start_smart_agent.bat` - быстрый запуск
- `SMART_AGENT_README.md` - полная документация (500+ строк)
- `FUZZY_SEARCH_FIX.md` - документация исправления fuzzy поиска

### Улучшения

#### ⚡ Исправлен catastrophic backtracking в fuzzy GREP
- **Проблема:** 5+ минут зависания при поиске
- **Причина:** fuzzy pattern создавался для всего предложения
- **Решение:** fuzzy только для отдельных ключевых слов (до 3)
- **Результат:** < 1 секунда вместо 5+ минут

#### 🚀 Обновлен start_rag.bat
- Теперь запускает SMART Agent по умолчанию
- Проверка Qwen3-30B-A3B модели
- Проверка Ultimate базы
- Порт 7861 (не конфликтует с обычным режимом)

### Технические детали

#### Архитектура SMART Agent
```
User Question
    ↓
Qwen3 Planning (анализ + выбор инструментов)
    ↓
Tool Execution (parallel grep/RAG/expand)
    ↓
Qwen3 Evaluation (достаточно ли информации?)
    ↓
[If NO] → Additional Tool Calls
    ↓
[If YES] → Final Synthesis
    ↓
Save to Memory (только финальный ответ!)
```

#### Системный промпт
```python
system_prompt = """Ты - экспертный ассистент по космоэнергетике с доступом к инструментам поиска.

Твоя задача:
1. Проанализировать вопрос
2. Решить какие инструменты нужны
3. Вызвать инструменты (можно несколько раз)
4. Синтезировать финальный ответ

Правила:
- grep_search: для конкретных имен
- rag_semantic_search: для концептуальных вопросов
- expand_query: если подозреваешь опечатки

ВАЖНО: Всегда проверяй достаточно ли информации!"""
```

#### Настройки производительности
- **max_context_tokens:** 16000 (было 8000)
- **summarize_threshold:** 11000 (было 5500)
- **max_short_memory:** 10 диалогов (было 5)
- **max_iterations:** 7 (новое)
- **temperature:** 0.3 (для точности)

### Сравнение режимов

| Параметр | Обычный режим | SMART Agent |
|---|---|---|
| Файл | rag_web_modern.py | rag_smart_qwen.py |
| Модель | Gemma-3-27B | Qwen3-30B-A3B |
| Порт | 7860 | 7861 |
| Режим | Вручную | Авто |
| Итераций | 1 | 1-7 |
| Контекст | 8000 | 16000 |
| Скорость | 2-5 сек | 5-15 сек |
| Качество | Хорошее | Максимальное |

### Как использовать

#### Быстрый старт SMART Agent
```bash
# Вариант 1: Двойной клик
start_rag.bat

# Вариант 2: Из командной строки
python rag_smart_qwen.py
```

Откройте: **http://localhost:7861**

#### Обычный режим (если нужен)
```bash
python rag_web_modern.py
```

Откройте: **http://localhost:7860**

### Примеры работы

#### Пример 1: Простой вопрос
```
User: "что такое Мектабу?"

Qwen3:
  Iteration 1: grep_search("Мектабу") → 12 результатов
  Iteration 2: rag_semantic_search("Мектабу описание") → 10 документов
  Final: подробный ответ

Время: 8 секунд
Инструментов: 2
```

#### Пример 2: Сложный вопрос
```
User: "какие расширяющие вопросы для Мектабу?"

Qwen3:
  Iteration 1: grep_search("Мектабу") → 12 результатов
  Iteration 2: expand_query("Мектабу") → ["Мектабу", "Мектаба", "Мектаб"]
  Iteration 3: rag_semantic_search("расширяющие вопросы") → 8 документов
  Evaluation: "мало контекста о Мектабу методике"
  Iteration 4: rag_semantic_search("Мектабу методика работа") → 6 документов
  Final: список расширяющих вопросов с контекстом

Время: 14 секунд
Инструментов: 4
```

### Требования

#### Железо
- **GPU:** RTX 3090 (24GB VRAM) - рекомендуется
- **RAM:** 32GB
- **Место:** 25GB (Qwen3 18.56GB + Ultimate база 2.2GB + буфер)

#### Модели
- **LLM:** Qwen3-30B-A3B (qwen/qwen3-30b-a3b-2507)
- **Embeddings:** intfloat/multilingual-e5-large
- **База:** chroma_db_ultimate

#### Программы
- **LM Studio** с Qwen3-30B-A3B
- **Python 3.10+**
- **CUDA 12.4+** (для GPU ускорения)

### Файлы проекта

```
advanced-rag-2025/
├── rag_smart_qwen.py           # SMART Agent (новое)
├── start_smart_agent.bat       # Запуск SMART (новое)
├── SMART_AGENT_README.md       # Документация (новое)
├── FUZZY_SEARCH_FIX.md         # Fuzzy fix (новое)
├── start_rag.bat               # Обновлен (SMART по умолчанию)
├── rag_web_modern.py           # Обычный режим (обновлен)
├── rag_advanced_memory.py      # Система памяти
├── create_ultimate_db.py       # Создание Ultimate базы
└── chroma_db_ultimate/         # Ultimate база данных
```

### Git Commits

```
e13677d Update start_rag.bat: Launch SMART Agent
e56b65c Add SMART RAG Agent with Qwen3 Function Calling
cd3cf47 Fix catastrophic backtracking in fuzzy GREP
```

### Известные проблемы

#### ⚠️ Qwen3 не вызывает инструменты
**Причина:** Модель не поддерживает function calling или неправильная версия

**Решение:**
1. Убедитесь что модель: `qwen/qwen3-30b-a3b-2507`
2. Обновите LM Studio до версии 0.3.0+
3. Проверьте логи: `smart_agent.log`

#### ⚠️ Ultimate база не найдена
**Решение:**
```bash
python create_ultimate_db.py
```

Подождите 15-20 минут.

#### ⚠️ Out of memory (CUDA)
**Решение:**
1. Закройте другие программы использующие GPU
2. Уменьшите `max_context_tokens` до 8000 в `rag_smart_qwen.py`
3. Используйте модель Q4_0 вместо Q4_K_M (меньше памяти)

### Roadmap

#### Версия 2.0 (планируется)
- [ ] Визуализация размышлений Qwen3 (граф вызовов)
- [ ] Дополнительные инструменты:
  - [ ] `summarize_document` - суммаризация документа
  - [ ] `compare_channels` - сравнение каналов
  - [ ] `find_contradictions` - поиск противоречий
- [ ] Настраиваемые стратегии поиска (агрессивная/консервативная)
- [ ] Кэширование результатов инструментов
- [ ] Мультиагентная система (несколько Qwen3 параллельно)
- [ ] Поддержка Gemma-3-27B для multimodal (картинки)

#### Версия 2.1 (планируется)
- [ ] Web UI для управления инструментами
- [ ] Metrics dashboard (статистика вызовов)
- [ ] A/B тестирование промптов
- [ ] Export результатов в Markdown/PDF
- [ ] Интеграция с Notion/Obsidian

### Благодарности

- **Google DeepMind** - Gemma-3 models
- **Alibaba Qwen Team** - Qwen3 models
- **LangChain** - RAG framework
- **ChromaDB** - vector database
- **Gradio** - web interface
- **LM Studio** - local LLM server

### License

MIT License

### Контакты

- **GitHub:** https://github.com/jetmil/advanced-rag-2025
- **Issues:** https://github.com/jetmil/advanced-rag-2025/issues

---

🤖 **Generated with Claude Code**
Co-Authored-By: Claude <noreply@anthropic.com>

---

## [2025-10-15] - Previous Updates

### Улучшения
- Hybrid search (GREP + RAG)
- Fuzzy search для GREP
- Ultimate база с multilingual-e5-large
- Увеличены лимиты контекста (8000 токенов)
- Glassmorphism UI
- Автосуммаризация диалогов

### Исправления
- GREP path detection
- Context overflow errors
- UI layout issues
- Database detection при загрузке
