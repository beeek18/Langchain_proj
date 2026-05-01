# agent.py
import json
import logging
from uuid import uuid4
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

logger = logging.getLogger(__name__)


def normalize_message_text(message) -> str:
    """Текст из ответа LLM: учитывает content_blocks (OpenAI v1 / gpt-5) и свойство .text."""
    try:
        blocks = getattr(message, "content_blocks", None) or []
    except Exception:
        blocks = []
    if blocks:
        chunks: list[str] = []
        for block in blocks or []:
            if not isinstance(block, dict):
                continue
            t = block.get("type")
            if t == "text" and block.get("text"):
                chunks.append(str(block["text"]))
            elif t == "reasoning" and (block.get("reasoning") or block.get("text")):
                chunks.append(str(block.get("reasoning") or block.get("text")))
            elif t == "non_standard" and isinstance(block.get("value"), dict):
                v = block["value"]
                if isinstance(v.get("text"), str):
                    chunks.append(v["text"])
        if chunks:
            return "\n".join(chunks).strip()
    try:
        tx = getattr(message, "text", None)
        if tx is not None and str(tx).strip():
            return str(tx).strip()
    except Exception:
        pass
    raw = getattr(message, "content", None)
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        chunks2: list[str] = []
        for part in raw:
            if isinstance(part, str):
                chunks2.append(part)
            elif isinstance(part, dict):
                t = part.get("type")
                if t == "text" and part.get("text"):
                    chunks2.append(str(part["text"]))
                elif part.get("text"):
                    chunks2.append(str(part["text"]))
                elif t == "reasoning" and part.get("reasoning"):
                    chunks2.append(str(part["reasoning"]))
                else:
                    for _k, _v in part.items():
                        if _k != "type" and isinstance(_v, str) and _v.strip():
                            chunks2.append(_v)
            else:
                chunks2.append(str(part))
        return "\n".join(chunks2).strip()
    return str(raw).strip()


def _tool_observations_blob(messages: list, max_chars: int = 14000) -> str:
    parts = [str(m.content) for m in messages if isinstance(m, ToolMessage) and m.content]
    blob = "\n\n---\n\n".join(parts)
    if len(blob) <= max_chars:
        return blob
    return blob[-max_chars:]


def _synthesize_from_tools(
    user_query: str,
    system_message: SystemMessage,
    messages: list,
    llm_plain,
) -> tuple[str, list]:
    """Когда модель с tool_calls вернула пустой финальный текст — собрать ответ из ToolMessage без инструментов."""
    blob = _tool_observations_blob(messages)
    if not blob.strip():
        return "", messages
    syn = llm_plain.invoke(
        [
            system_message,
            HumanMessage(
                content=(
                    f"Запрос пользователя: {user_query}\n\n"
                    f"Ниже сырые данные из инструментов (рецепты). Составь понятный ответ на русском: "
                    f"названия, время, калории на 100 г, ингредиенты, ссылки. Не выдумывай рецептов сверх того, что есть в данных.\n\n"
                    f"{blob}"
                )
            ),
        ]
    )
    text = normalize_message_text(syn)
    if not text:
        return "", messages
    if messages and isinstance(messages[-1], AIMessage) and not normalize_message_text(messages[-1]):
        messages.pop()
    messages.append(AIMessage(content=text))
    return text, messages


async def run_agent_bot(
    user_query: str,
    llm_with_tools,
    tools_map: dict,
    system_message: SystemMessage,
    session_id: str | None = None,
    history: list | None = None,
    max_iters: int = 10,
    llm_plain=None,
) -> tuple[str, str, list]:
    """
    Возвращает (session_id, ответ_агента, history).
    """
    if session_id is None:
        session_id = f"sess-{str(uuid4())[:6]}"

    messages = list(history or [system_message])
    messages.append(HumanMessage(content=user_query))
    used_calls: set = set()

    for step in range(1, max_iters + 1):
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        logger.info(
            "run_agent_bot step=%s tool_calls=%s content_len=%s",
            step,
            len(ai_msg.tool_calls) if ai_msg.tool_calls else 0,
            len(normalize_message_text(ai_msg)),
        )

        if not ai_msg.tool_calls:
            text = normalize_message_text(ai_msg)
            if not text and any(isinstance(m, ToolMessage) for m in messages):
                logger.warning(
                    "run_agent_bot: пустой финальный ответ при наличии ToolMessage, шаг=%s — синтез через llm_plain",
                    step,
                )
                if llm_plain is not None:
                    text, messages = _synthesize_from_tools(
                        user_query, system_message, messages, llm_plain
                    )
                    if text:
                        logger.info("run_agent_bot: синтез из ToolMessage, len=%s", len(text))
                if not text and llm_plain is not None:
                    blob = _tool_observations_blob(messages)
                    if blob.strip():
                        syn2 = llm_plain.invoke(
                            [
                                HumanMessage(
                                    content=(
                                        "Ниже данные о рецептах. Кратко и по делу на русском перечисли каждый: "
                                        "название, время, калории, ингредиенты, ссылка.\n\n"
                                        f"{blob[:12000]}"
                                    )
                                )
                            ]
                        )
                        text = normalize_message_text(syn2)
                        if text and messages and isinstance(messages[-1], AIMessage):
                            if not normalize_message_text(messages[-1]):
                                messages.pop()
                            messages.append(AIMessage(content=text))
                if (
                    not text
                    and messages
                    and isinstance(messages[-1], AIMessage)
                    and not normalize_message_text(messages[-1])
                ):
                    messages.pop()
            if not text:
                blob_fb = _tool_observations_blob(messages)
                if blob_fb.strip():
                    logger.warning(
                        "run_agent_bot: модель вернула пустой текст — показываю сырой вывод инструментов, len=%s",
                        len(blob_fb),
                    )
                    text = "Данные из поиска (текст от модели не пришёл):\n\n" + blob_fb[:14000]
                    if messages and isinstance(messages[-1], AIMessage):
                        if not normalize_message_text(messages[-1]):
                            messages.pop()
                    messages.append(AIMessage(content=text))
                else:
                    logger.warning(
                        "run_agent_bot: пустой текст после шага %s, tool_calls=%s, raw_content=%r",
                        step,
                        getattr(ai_msg, "tool_calls", None),
                        getattr(ai_msg, "content", None),
                    )
                    text = (
                        "Не удалось сформулировать ответ и нет данных от инструментов в этой сессии. "
                        "Попробуй снова запросить блюдо или укажи ID рецепта из прошлого ответа."
                    )
            return session_id, text, messages

        for tc in ai_msg.tool_calls:
            t_name, t_args, t_id = tc["name"], tc["args"], tc["id"]

            sig = (t_name, json.dumps(t_args, sort_keys=True))
            if sig in used_calls and t_name != "search_recipes":
                observation = "Данные уже получены ранее. Используй результаты предыдущего вызова для ответа."
            else:
                used_calls.add(sig)
                try:
                    observation = tools_map[t_name].invoke(t_args) if t_name in tools_map else f"Unknown tool: {t_name}"
                except Exception as err:
                    logger.exception("tool %s failed args=%s", t_name, t_args)
                    observation = f"Ошибка инструмента {t_name}: {err}"

            messages.append(ToolMessage(content=str(observation), tool_call_id=t_id))

    logger.warning("run_agent_bot: достигнут max_iters=%s", max_iters)
    return session_id, "Не удалось найти ответ.", messages