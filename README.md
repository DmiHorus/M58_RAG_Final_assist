# M58 RAG Final Assist

RAG-ассистент с двумя режимами работы: через **OpenAI API** и через **GigaChat API** (Сбер). Проект включает кеширование ответов, векторный поиск в ChromaDB, оценку качества через RAGAS и базу знаний на основе архитектурного документа.

## Возможности

- **Retrieval-Augmented Generation** — ответы на основе релевантных фрагментов из базы знаний
- **Кеширование** — SQLite-кеш для повторных запросов (снижение latency и затрат на API)
- **Семантический поиск** — OpenAI Embeddings (`text-embedding-3-small`) + ChromaDB
- **Markdown-aware chunking** — разбиение `rag_architecture.md` с учётом заголовков и разделителей
- **Оценка качества** — метрики Faithfulness и Context Precision через RAGAS
- **Два LLM-провайдера** — OpenAI (`assistant_api`) и GigaChat (`assistant_giga`)

## Структура проекта

```
M58_RAG_Final_assist/
├── assistant_api/              # RAG на OpenAI
│   ├── app.py                  # Консольное приложение
│   ├── rag_pipeline.py         # Основной pipeline
│   ├── vector_store.py         # ChromaDB + embeddings
│   ├── cache.py                # SQLite-кеш
│   ├── evaluate_ragas.py       # Оценка качества (RAGAS)
│   └── data/
│       └── rag_architecture.md # База знаний
├── assistant_giga/             # RAG на GigaChat
│   ├── app.py
│   ├── rag_pipeline.py
│   ├── vector_store.py
│   ├── cache.py
│   ├── gigachat_client.py
│   └── data/
│       └── rag_architecture.md
├── requirements.txt
├── .env.example
└── README.md
```

## Требования

- Python **3.11**
- API-ключ [OpenAI](https://platform.openai.com/) — для `assistant_api`
- Ключ [GigaChat API](https://developers.sber.ru/portal/products/gigachat-api) — для `assistant_giga`

## Установка

```bash
# Клонирование репозитория
git clone https://github.com/DmiHorus/M58_RAG_Final_assist.git
cd M58_RAG_Final_assist

# Создание виртуального окружения
python3.11 -m venv venv_py311
source venv_py311/bin/activate   # macOS / Linux
# .\venv_py311\Scripts\activate  # Windows

# Установка зависимостей
pip install --upgrade pip
pip install -r requirements.txt

# Настройка переменных окружения
cp .env.example .env
# Отредактируйте .env — укажите свои API-ключи
```

## Конфигурация (.env)

| Переменная | Описание |
|---|---|
| `OPENAI_API_KEY` | Ключ OpenAI (обязательно для `assistant_api`) |
| `OPENAI_MODEL` | Модель генерации (по умолчанию `gpt-4o-mini`) |
| `TEMPERATURE` | Температура генерации (по умолчанию `0.7`) |
| `GIGACHAT_AUTH_KEY` | Basic-токен GigaChat (для `assistant_giga`) |
| `GIGACHAT_RQUID` | UUID v4 для OAuth-запросов GigaChat |
| `GIGACHAT_SCOPE` | Scope API (по умолчанию `GIGACHAT_API_PERS`) |
| `RAGAS_LLM_MODEL` | Модель для метрик RAGAS (по умолчанию = `OPENAI_MODEL`) |

## Запуск

### OpenAI-ассистент

```bash
cd assistant_api
python app.py
```

Команды в консоли:
- `stats` — статистика кеша и векторного хранилища
- `clear` — очистка кеша
- `exit` / `quit` — выход

### GigaChat-ассистент

```bash
cd assistant_giga
python app.py
```

### Оценка качества (RAGAS)

```bash
cd assistant_api
python evaluate_ragas.py
```

Скрипт прогоняет 5 тестовых вопросов по `rag_architecture.md` и выводит метрики **Faithfulness** и **Context Precision**.

## Архитектура pipeline

```
Запрос пользователя
       │
       ▼
  ┌─────────┐    hit     ┌──────────┐
  │  Cache  │───────────►│  Ответ   │
  └────┬────┘            └──────────┘
       │ miss
       ▼
  ┌─────────────┐
  │ Vector Store│  ← ChromaDB + OpenAI Embeddings
  │  (top_k=3)  │
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ Prompt + LLM│  ← OpenAI / GigaChat
  └──────┬──────┘
         ▼
  ┌─────────┐
  │  Cache  │ → Ответ
  └─────────┘
```

## Параметры vector store

| Параметр | Значение |
|---|---|
| Модель эмбеддингов | `text-embedding-3-small` (1536 dim) |
| Chunk size | 600 символов |
| Overlap | 120 символов |
| Метрика ChromaDB | cosine |
| top_k | 3 |

## База знаний

Источник данных — `data/rag_architecture.md`: архитектурное описание RAG-системы для базы бизнес-процессных сценариев (hybrid retrieval, ingestion pipeline, хранилища знаний, генерация ответов).

При первом запуске документ автоматически разбивается на чанки и индексируется в ChromaDB.

## Лицензия

Учебный проект. Используйте на своё усмотрение.
