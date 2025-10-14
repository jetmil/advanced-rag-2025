# 🚀 LM Studio Server - Руководство

## 2 способа запуска сервера LM Studio

---

## 🖥️ Способ 1: Через GUI (графический интерфейс)

### Пошаговая инструкция:

1. **Откройте LM Studio**

2. **Найдите вкладку сервера:**
   - Слева в боковой панели найдите иконку ↔️ или текст **"Local Server"** / **"Developer"**
   - Кликните на эту вкладку

3. **Настройте сервер:**
   ```
   ┌─────────────────────────────────┐
   │ Local Inference Server          │
   ├─────────────────────────────────┤
   │ Model to load:                  │
   │ [Select model ▼]                │
   │                                 │
   │ Select: qwen/qwen3-30b-a3b-2507 │
   │                                 │
   │ Port: [1234]                    │
   │ GPU Layers: [Auto/Max]          │
   │                                 │
   │ [🟢 Start Server]               │
   └─────────────────────────────────┘
   ```

4. **Нажмите "Start Server"**

5. **Сервер запущен!**
   - Адрес: `http://localhost:1234`
   - Индикатор: зеленый кружок 🟢

---

## ⚡ Способ 2: Через CLI (фоновый режим БЕЗ GUI)

Если вы хотите запустить сервер **без открытия GUI LM Studio** (headless mode):

### Быстрый запуск:

**Просто запустите bat-файл:**
```bash
start_lmstudio_server.bat
```

### Или вручную через команды:

```bash
# 1. Загрузить модель
lms load qwen/qwen3-30b-a3b-2507 --gpu max --yes

# 2. Запустить сервер
lms server start --port 1234 --cors
```

---

## 📁 Созданные файлы для управления сервером

### 1. `start_lmstudio_server.bat`
**Запуск сервера через CLI**
```bash
# Двойной клик или в консоли:
start_lmstudio_server.bat
```
- Загружает Qwen3-30B в память
- Запускает сервер на порту 1234
- Максимальное использование GPU

### 2. `check_lmstudio_status.bat`
**Проверка статуса**
```bash
check_lmstudio_status.bat
```
Показывает:
- Какие модели загружены
- Работает ли сервер
- На каком порту

### 3. `stop_lmstudio_server.bat`
**Остановка сервера**
```bash
stop_lmstudio_server.bat
```
Корректно останавливает сервер

---

## 🔍 Проверка работы сервера

### Метод 1: Через bat-файл
```bash
check_lmstudio_status.bat
```

### Метод 2: Через браузер
Откройте: `http://localhost:1234/v1/models`

Должно вернуть JSON с моделями:
```json
{
  "data": [
    {
      "id": "qwen/qwen3-30b-a3b-2507",
      ...
    }
  ]
}
```

### Метод 3: Через curl
```bash
curl http://localhost:1234/v1/models
```

### Метод 4: Через Python
```python
import requests
response = requests.get("http://localhost:1234/v1/models")
print(response.json())
```

---

## 🎯 Полный процесс: от запуска до работы с RAG

### Вариант A: GUI + RAG

```bash
# 1. Запустите LM Studio (GUI)
#    - Откройте приложение
#    - Local Server → выберите модель → Start Server

# 2. Запустите RAG веб-интерфейс
python rag_web_interface.py

# 3. Откройте браузер
#    http://localhost:7860
```

### Вариант B: CLI + RAG (без GUI)

```bash
# 1. Запустите сервер через CLI (в отдельном окне)
start_lmstudio_server.bat

# 2. Запустите RAG веб-интерфейс (в другом окне)
python rag_web_interface.py

# 3. Откройте браузер
#    http://localhost:7860
```

---

## 🛠️ Команды lms CLI

### Основные команды:

```bash
# Список всех моделей
lms ls

# Список загруженных моделей
lms ps

# Загрузить модель
lms load qwen/qwen3-30b-a3b-2507 --gpu max

# Выгрузить модель
lms unload qwen/qwen3-30b-a3b-2507

# Статус сервера
lms server status

# Запуск сервера
lms server start --port 1234

# Остановка сервера
lms server stop

# Чат с моделью в консоли
lms chat
```

### Параметры загрузки модели:

```bash
# Максимальное использование GPU
lms load <model> --gpu max

# Частичное использование GPU (50%)
lms load <model> --gpu 0.5

# Без GPU (только CPU)
lms load <model> --gpu off

# С автовыгрузкой через 300 секунд
lms load <model> --ttl 300

# Кастомная длина контекста
lms load <model> --context-length 8192
```

---

## 🔧 Troubleshooting

### ❌ Ошибка: "Server already running"

**Решение:**
```bash
# Остановите текущий сервер
lms server stop

# Или используйте другой порт
lms server start --port 1235
```

### ❌ Ошибка: "Cannot connect to LM Studio"

**Проверьте:**
1. Запущен ли сервер?
   ```bash
   lms server status
   ```

2. Правильный ли порт?
   ```bash
   curl http://localhost:1234/v1/models
   ```

3. Модель загружена?
   ```bash
   lms ps
   ```

### ❌ Ошибка: "Model not found"

**Решение:**
```bash
# Проверьте список моделей
lms ls

# Загрузите нужную модель
lms load qwen/qwen3-30b-a3b-2507 --yes
```

### ⚠️ Медленная работа

**Причины:**
- Модель не на GPU
- Другие приложения используют GPU
- Недостаточно VRAM

**Решение:**
```bash
# Перезагрузите с максимальным GPU
lms unload qwen/qwen3-30b-a3b-2507
lms load qwen/qwen3-30b-a3b-2507 --gpu max
```

---

## 📊 Мониторинг ресурсов

### Проверка использования GPU:

**Через nvidia-smi:**
```bash
nvidia-smi
```

**Мониторинг в реальном времени:**
```bash
nvidia-smi -l 1
```

**Ожидаемое использование для Qwen3-30B:**
- VRAM: ~18-20 GB
- GPU Load: 30-90% (зависит от запросов)

---

## 🚀 Автозапуск сервера при старте Windows

### Создание задачи планировщика:

1. Откройте **Планировщик заданий** (Task Scheduler)
2. **Создать задачу** → **Общие**:
   - Имя: "LM Studio Server"
   - Выполнять при входе в систему
3. **Триггеры** → Добавить:
   - При входе в систему
4. **Действия** → Добавить:
   - Программа: `C:\Users\PC\start_lmstudio_server.bat`
5. **Сохранить**

Теперь сервер будет запускаться автоматически!

---

## 💡 Полезные советы

### 1. Работа в фоне
Сервер CLI работает в фоне - можете закрыть окно консоли после запуска (если не нужны логи).

### 2. Несколько моделей одновременно
```bash
# Загрузить вторую модель
lms load другая-модель --identifier model2
```

### 3. Удаленный доступ
Если нужен доступ с других компьютеров в сети:
```bash
lms server start --port 1234 --cors
```
Затем используйте `http://<your-ip>:1234`

### 4. Логирование
```bash
# Подробные логи
lms server start --verbose

# Без логов
lms server start --quiet
```

---

## 📖 API Documentation

LM Studio использует OpenAI-совместимый API:

**Base URL:** `http://localhost:1234/v1`

**Endpoints:**
- `/v1/models` - список моделей
- `/v1/chat/completions` - чат
- `/v1/completions` - текстовая генерация
- `/v1/embeddings` - embeddings (если модель поддерживает)

**Пример запроса (curl):**
```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen/qwen3-30b-a3b-2507",
    "messages": [{"role": "user", "content": "Привет!"}],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

**Пример запроса (Python):**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="qwen/qwen3-30b-a3b-2507",
    messages=[{"role": "user", "content": "Привет!"}],
    temperature=0.7
)

print(response.choices[0].message.content)
```

---

## ✅ Чеклист перед запуском RAG

- [ ] LM Studio сервер запущен (GUI или CLI)
- [ ] Модель Qwen3-30B загружена
- [ ] Сервер отвечает на `http://localhost:1234/v1/models`
- [ ] Есть свободно ~18GB VRAM на GPU
- [ ] Установлены все Python библиотеки
- [ ] Текстовый файл находится по пути

Если все ✅ - запускайте RAG!

---

**Создано:** Октябрь 2025
**Для:** RTX 3090, LM Studio, Windows
