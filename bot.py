import asyncio
import os
import torch
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import CommandStart
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from toolkit import RecipeToolkit
from guard import process_query
from agent import run_agent_bot

load_dotenv()

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

llm = ChatOpenAI(model="gpt-5-nano", api_key=os.environ["OPENAI_API_KEY"], max_tokens=2000)

recipe_toolkit = RecipeToolkit(vector_store=vector_store)
tools = recipe_toolkit.get_tools()
tools_map = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

SYSTEM_MESSAGE = SystemMessage(content="""Ты — кулинарный помощник. Помогаешь пользователям найти рецепты.
Логика работы — строго по шагам:
1. ВСЕГДА начинай с search_recipes(query) — только название блюда, без фильтров
2. Если результаты РЕЛЕВАНТНЫ:
   - Если есть фильтры (время, калории, БЖУ, аллергены) — вызови search_recipes_with_filters
   - Если фильтров нет — сразу отвечай пользователю
3. Если результаты НЕРЕЛЕВАНТНЫ или база пуста — вызови scrape_and_save_recipe ОДИН РАЗ
4. После scrape_and_save_recipe — НЕ иди снова в search_recipes, используй результаты scrape напрямую
5. Если пользователь хочет похожие — используй find_similar с ID из предыдущего ответа

Важно про запросы:
- Всегда используй конкретное название блюда на русском: 'борщ', 'тирамису', 'плов'
- Если пользователь написал размыто — сам выбери конкретное блюдо
- Никогда не передавай абстрактные слова в тулы

Формат ответа — ВСЕГДА включай:
- Название, время готовки, калории на 100г, ингредиенты, ссылку

Отвечай на русском языке.
""")

# ── Хранилище сессий ───────────────────────
sessions: dict[int, tuple[str, list]] = {}  # user_id → (session_id, history)

bot = Bot(token=os.environ["TELEGRAM_BOT_TOKEN"])
dp = Dispatcher()


@dp.message(CommandStart())
async def start(message: Message):
    sessions.pop(message.from_user.id, None)  # сбрасываем историю
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

    # Guard
    status, result = process_query(user_input, llm)
    if status == "scam":
        await message.answer(result)
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
        )
        sessions[user_id] = (session_id, history)
        await message.answer(answer, parse_mode="Markdown")
    except Exception as e:
        await message.answer(f"❌ Ошибка: {e}")


async def main():
    print("🤖 Бот запущен")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())