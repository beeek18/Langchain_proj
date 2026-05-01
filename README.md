# 🍳 Recipe Agent

RAG-агент для поиска рецептов с защитой от prompt injection.

## Архитектура

```
Запрос пользователя
        │
        ▼
remove_system_tokens()     ← убираем инъекции токенов
        │
        ▼
Guard LLM                  ← роутим: recipe или scam?
        │
   ┌────┴────┐
scam       recipe
  │            │
токсичный   run_agent()
  ответ          │
            search_recipes()        ← поиск без фильтров
                  │
          ┌───────┴───────┐
       найдено        нет / нерелевантно
          │                  │
     есть фильтры?    scrape_and_save_recipe()
          │            парсим povarenok.ru
     ┌────┴────┐        сохраняем в Qdrant
    да        нет
     │         │
search_      ответ
with_       пользователю
filters()
```

## Стек

- **LLM** — OpenAI GPT-4o-mini
- **Векторная БД** — Qdrant (локально, папка `./qdrant_db`)
- **Embeddings** — `intfloat/multilingual-e5-small` (HuggingFace)
- **Фреймворк** — LangChain
- **Скрапер** — Playwright (Chromium)
- **Трейсинг** — LangSmith
- **Источник рецептов** — povarenok.ru

## Структура проекта

```
├── main.ipynb          # Точка входа — агент и run_agent()
├── toolkit.py          # LangChain тулкит (4 тула)
├── scraper.py          # Playwright скрапер povarenok.ru
├── guard.py            # Guard LLM + декоратор + стоп-токены
├── seed_db.py          # Наполнение БД популярными рецептами
├── qdrant_db/          # Локальная векторная БД
└── .env                # Ключи API
```

## Тулы агента

| Тул | Описание |
|-----|----------|
| `search_recipes` | Семантический поиск в Qdrant без фильтров |
| `search_recipes_with_filters` | Поиск с фильтрами по времени, калориям, БЖУ, аллергенам |
| `find_similar` | Похожие рецепты по ID через векторное сходство |
| `scrape_and_save_recipe` | Парсит povarenok.ru и сохраняет в Qdrant |

## Фильтры поиска

- `max_cook_time` / `min_cook_time` — время готовки в минутах
- `max_calories` / `min_calories` — калории на 100г
- `max_protein` / `min_protein` — белки на 100г
- `max_fat` / `min_fat` — жиры на 100г
- `max_carbs` / `min_carbs` — углеводы на 100г
- `exclude_ingredients` — исключить ингредиенты (корень слова: `яйц`, `молок`)

## Быстрый старт

### 1. Установка зависимостей

```bash
uv add qdrant-client langchain-huggingface langchain-openai langchain-qdrant \
       playwright langsmith python-dotenv nest-asyncio sentence-transformers tenacity
uv run playwright install chromium
```

### 2. Настройка `.env`

```env
OPENAI_API_KEY=sk-...
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=ls__...
LANGSMITH_PROJECT=recipe-agent
```

### 3. Наполнение БД

```bash
uv run seed_db.py
```

Скрипт загружает 300 популярных блюд по категориям. Помнит где остановился — можно прерывать и продолжать.

### 4. Запуск

Открой `main.ipynb` и запускай ячейки сверху вниз.

### Telegram-бот

1. Создай бота у [@BotFather](https://t.me/BotFather), скопируй токен.
2. В `.env` добавь `TELEGRAM_BOT_TOKEN=<токен>` и проверь `OPENAI_API_KEY`.
3. Убедись, что есть `./qdrant_db` (как минимум один раз выполни `uv run seed_db.py` или пользуйся приложением, которое уже наполнило БД).
4. Запуск:

```bash
uv run python bot.py
```

Логи пишутся в stderr (статус guard, ошибки `chat_reply` и агента). Длинные ответы режутся по ~4000 символов; если разметка Markdown невалидна, сообщение уходит без `parse_mode`.

## Примеры запросов

```python
# Простой поиск
sid, hist = run_agent("хочу борщ", ...)

# С фильтрами
sid, hist = run_agent("хочу плов за час и не более 200 ккал", ...)

# С аллергией
sid, hist = run_agent("что-нибудь без яиц и молока", ...)

# По кухне
sid, hist = run_agent("хочу итальянское", ...)

# По случаю
sid, hist = run_agent("что приготовить на день рождения", ...)

# Продолжение диалога
sid, hist = run_agent("найди похожие на первый", ..., history=hist)
```

## Защита от атак

**Стоп-токены** — удаляются системные токены всех популярных LLM (`[INST]`, `<|im_start|>`, `Human:` и др.) и паттерны prompt injection (`IGNORE ALL PREVIOUS`, `OVERRIDE INSTRUCTIONS` и др.)

**Guard LLM** — отдельная модель классифицирует запрос:
- `recipe` → передаём агенту
- `scam` → возвращаем токсичный ответ

Декоратор `@guard_decorator(llm)` применяется поверх `run_agent` — вредоносные запросы не доходят до основного агента.

## Устойчивость к сбоям

**Tenacity** — декоратор `@retry` на `parse_page` делает 3 попытки с паузой 2 секунды между ними при таймаутах или сетевых ошибках.

**Защита от зацикливания** — `used_calls` блокирует повторные вызовы одного тула с теми же аргументами.

**Продолжение после сбоя** — `seed_db.py` сохраняет прогресс в `seeded.txt` и продолжает с места остановки.


# В БД 2511 рецептов