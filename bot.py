import asyncio
import logging
import os
import sys
import torch
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import Message
from aiogram.filters import CommandStart
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from toolkit import RecipeToolkit
from guard import process_query, chat_reply
from agent import run_agent_bot

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
    force=True,
)
log = logging.getLogger("telegram_bot")

TELEGRAM_MAX_LEN = 4000


def _chunk_for_telegram(text: str, limit: int = TELEGRAM_MAX_LEN) -> list[str]:
    if len(text) <= limit:
        return [text]
    return [text[i : i + limit] for i in range(0, len(text), limit)]


async def _answer_user(message: Message, text: str, *, use_markdown: bool) -> None:
    for part in _chunk_for_telegram(text):
        if use_markdown:
            try:
                await message.answer(part, parse_mode="Markdown")
            except TelegramBadRequest:
                await message.answer(part)
        else:
            await message.answer(part)


def _require_env(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise RuntimeError(f"Задай переменную окружения {name} (см. .env.example)")
    return value


# ── Инициализация ──────────────────────────
client = QdrantClient(path="./qdrant_db")
if not client.collection_exists("recipes"):
    client.create_collection("recipes", vectors_config=VectorParams(size=384, distance=Distance.COSINE))

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
embeddings_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)
vector_store = QdrantVectorStore(client=client, collection_name="recipes", embedding=embeddings_model)

llm = ChatOpenAI(model="gpt-5-nano", api_key=_require_env("OPENAI_API_KEY"), max_tokens=2000)

recipe_toolkit = RecipeToolkit(vector_store=vector_store)
tools = recipe_toolkit.get_tools()
tools_map = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

SYSTEM_MESSAGE = SystemMessage(content="""Ты — кулинарный помощник. Помогаешь пользователям найти рецепты.

Логика работы — строго по шагам:
1. Если пользователь просит похожие / «ещё варианты» / «на твой вкус» после уже показанного рецепта — вызови find_similar(recipe_id, limit) с ID из последнего ответа или из ToolMessage в истории (limit до 10). Не вызывай search_recipes, если уже есть подходящий recipe_id в контексте.
2. Если в запросе есть ограничения по времени готовки, калориям на 100 г, БЖУ или аллергенам — и при этом НЕТ конкретного названия блюда — сразу вызови search_recipes_with_filters с подходящими max_cook_time / max_calories и т.д. и коротким обобщённым query на русском («лёгкое блюдо», «быстро приготовить», «низкокалорийное»). Не подставляй случайный десерт из прошлого контекста.
3. Иначе для нового блюда с названием начинай с search_recipes(query) — только название блюда, без фильтров.
4. Если результаты search_recipes релевантны и пользователь (или запрос) требует фильтры — вызови search_recipes_with_filters с тем же смыслом query и фильтрами.
5. Если фильтров нет после search_recipes — сразу отвечай пользователю.
6. Если результаты НЕРЕЛЕВАНТНЫ или база пуста — вызови scrape_and_save_recipe ОДИН РАЗ
7. После scrape_and_save_recipe — НЕ иди снова в search_recipes, используй результаты scrape напрямую

Важно про запросы:
- Для обычного поиска используй конкретное название блюда на русском: 'борщ', 'тирамису', 'плов'
- Если запрос размытый, но без числовых ограничений — сам выбери конкретное блюдо для search_recipes
- В ответе пользователю честно сравни рецепты с его ограничениями (если калорийность или время не подходят — скажи об этом)

Формат ответа — ВСЕГДА включай:
- Название, время готовки, калории на 100г, ингредиенты, ссылку

Отвечай на русском языке.
""")

# ── Хранилище сессий ───────────────────────
sessions: dict[int, tuple[str, list]] = {}  # user_id → (session_id, history)
chat_transcripts: dict[int, list[dict]] = {}  # user_id → [{role, content}, ...] для chat_reply

bot = Bot(token=_require_env("TELEGRAM_BOT_TOKEN"))
dp = Dispatcher()


@dp.message(CommandStart())
async def start(message: Message):
    uid = message.from_user.id
    sessions.pop(uid, None)
    chat_transcripts.pop(uid, None)
    await message.answer(
        "👨‍🍳 Привет! Я кулинарный помощник.\n\n"
        "Спроси меня про любое блюдо и я найду рецепт!\n"
        "Например: *хочу борщ* или *что приготовить из курицы*",
        parse_mode="Markdown"
    )


@dp.message(F.text)
async def handle_message(message: Message):
    user_id = message.from_user.id
    user_input = message.text
    transcript = chat_transcripts.setdefault(user_id, [])
    transcript.append({"role": "user", "content": user_input})

    # Guard
    status, result = process_query(user_input, llm)
    log.info("turn user_id=%s status=%s preview=%r", user_id, status, (result or "")[:120])
    if status == "scam":
        transcript.append({"role": "assistant", "content": result})
        await _answer_user(message, result, use_markdown=False)
        return

    if status == "chat":
        try:
            answer = chat_reply(llm, session_messages=transcript)
        except Exception:
            log.exception("chat_reply failed")
            answer = "Ошибка при генерации ответа — смотри лог в терминале."
        if not (answer or "").strip():
            answer = "Пустой ответ модели — попробуй переформулировать запрос."
        transcript.append({"role": "assistant", "content": answer})
        await _answer_user(message, answer, use_markdown=False)
        return

    await message.answer("🔍 Ищу рецепт...")

    # Получаем историю сессии
    session_id, history = sessions.get(user_id, (None, [SYSTEM_MESSAGE]))

    try:
        session_id, answer, history = await run_agent_bot(
            user_query=result,
            llm_with_tools=llm_with_tools,
            tools_map=tools_map,
            system_message=SYSTEM_MESSAGE,
            session_id=session_id,
            history=history,
            llm_plain=llm,
        )
        sessions[user_id] = (session_id, history)
        if not (answer or "").strip():
            answer = "Пустой ответ — см. логи в терминале."
        transcript.append({"role": "assistant", "content": answer})
        await _answer_user(message, answer, use_markdown=True)
    except Exception as e:
        log.exception("run_agent_bot failed")
        await message.answer(f"❌ Ошибка агента: {e}")


async def main():
    log.info("Бот запущен (polling)")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())