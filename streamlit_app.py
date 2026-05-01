# streamlit_app.py
import logging
import sys

import streamlit as st
import os
import torch
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from toolkit import RecipeToolkit, run_async
from guard import process_query, chat_reply
from agent import run_agent_bot

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
    force=True,
)
log = logging.getLogger("streamlit_app")

st.set_page_config(page_title="Recipe Agent", page_icon="🍳")
st.title("Кулинарный помощник")
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
1. Если пользователь просит похожие / «ещё варианты» / «на твой вкус» после уже показанного рецепта — вызови find_similar(recipe_id, limit) с ID из последнего ответа или из ToolMessage в истории (limit до 10). Не вызывай search_recipes, если уже есть подходящий recipe_id в контексте.
2. Иначе для нового блюда ВСЕГДА начинай с search_recipes(query) — только название блюда, без фильтров.
3. Если результаты РЕЛЕВАНТНЫ:
   - Если есть фильтры (время, калории, БЖУ, аллергены) — вызови search_recipes_with_filters
   - Если фильтров нет — сразу отвечай пользователю
4. Если результаты НЕРЕЛЕВАНТНЫ или база пуста — вызови scrape_and_save_recipe ОДИН РАЗ
5. После scrape_and_save_recipe — НЕ иди снова в search_recipes, используй результаты scrape напрямую

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
    log.info("turn: status=%s result_preview=%r", status, (result or "")[:120])
    if status == "scam":
        st.session_state.messages.append({"role": "assistant", "content": result})
        with st.chat_message("assistant"):
            st.markdown(result)
    elif status == "chat":
        with st.chat_message("assistant"):
            with st.spinner("Пишу ответ..."):
                try:
                    answer = chat_reply(llm, session_messages=st.session_state.messages)
                except Exception:
                    log.exception("chat_reply failed")
                    answer = "Ошибка при генерации ответа — смотри терминал, где запущен `streamlit run`."
            if not (answer or "").strip():
                answer = "Пустой ответ модели — попробуй переформулировать запрос."
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.markdown(answer)
    else:
        with st.chat_message("assistant"):
            with st.spinner("🔍 Ищу рецепт..."):
                try:
                    session_id, answer, history = run_async(run_agent_bot(
                        user_query=result,
                        llm_with_tools=llm_with_tools,
                        tools_map=tools_map,
                        system_message=SYSTEM_MESSAGE,
                        session_id=st.session_state.session_id,
                        history=st.session_state.history,
                        llm_plain=llm,
                    ))
                except Exception:
                    log.exception("run_agent_bot failed")
                    session_id, answer, history = (
                        st.session_state.session_id,
                        "Ошибка агента — детали в терминале (stderr), где запущен Streamlit.",
                        st.session_state.history,
                    )
                st.session_state.session_id = session_id
                st.session_state.history = history
                if not (answer or "").strip():
                    answer = "Пустой ответ — см. логи в терминале."
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.markdown(answer)