# agent.py
import json
from uuid import uuid4
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


async def run_agent_bot(
    user_query: str,
    llm_with_tools,
    tools_map: dict,
    system_message: SystemMessage,
    session_id: str | None = None,
    history: list | None = None,
    max_iters: int = 10,
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

        if not ai_msg.tool_calls:
            return session_id, ai_msg.content, messages

        for tc in ai_msg.tool_calls:
            t_name, t_args, t_id = tc["name"], tc["args"], tc["id"]

            sig = (t_name, json.dumps(t_args, sort_keys=True))
            if sig in used_calls and t_name != "search_recipes":
                observation = "Данные уже получены ранее. Используй результаты предыдущего вызова для ответа."
            else:
                used_calls.add(sig)
                observation = tools_map[t_name].invoke(t_args) if t_name in tools_map else f"Unknown tool: {t_name}"

            messages.append(ToolMessage(content=str(observation), tool_call_id=t_id))

    return session_id, "Не удалось найти ответ.", messages