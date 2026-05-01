# streamlit_app.py
import streamlit as st
import asyncio
import os
import torch
import nest_asyncio
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from toolkit import RecipeToolkit
from guard import process_query
from agent import run_agent_bot

nest_asyncio.apply()
load_dotenv()

st.set_page_config(page_title="🍳 Recipe Agent", page_icon="🍳")
st.title("🍳 Кулинарный помощник")
st.caption("Спроси меня про любое блюдо и я найду рецепт!")


# ── Инициализация (один раз) ───────────────
@st.cache_resource
def init():
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
    return llm, llm_with_tools, tools_map


llm, llm_with_tools, tools_map = init()

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

# ── История чата ───────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = [SYSTEM_MESSAGE]
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Отображаем историю
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Ввод пользователя ──────────────────────
if user_input := st.chat_input("Например: хочу борщ"):
    # Показываем сообщение пользователя
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Guard
    status, result = process_query(user_input, llm)
    if status == "scam":
        st.session_state.messages.append({"role": "assistant", "content": result})
        with st.chat_message("assistant"):
            st.markdown(result)
    else:
        with st.chat_message("assistant"):
            with st.spinner("🔍 Ищу рецепт..."):
                session_id, answer, history = asyncio.run(run_agent_bot(
                    user_query=result,
                    llm_with_tools=llm_with_tools,
                    tools_map=tools_map,
                    system_message=SYSTEM_MESSAGE,
                    session_id=st.session_state.session_id,
                    history=st.session_state.history,
                ))
                st.session_state.session_id = session_id
                st.session_state.history = history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.markdown(answer)