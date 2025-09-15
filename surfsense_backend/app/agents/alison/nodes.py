import json
from typing import Any, List
from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamWriter
from .state import AlisonState
from .prompts import (
    get_alison_system_prompt,
    get_problem_identification_prompt,
    get_escalation_prompt,
)
from langchain_core.messages import SystemMessage, HumanMessage
from app.services.llm_service import get_user_fast_llm
import os
import pathlib

async def identify_problem(state: AlisonState, config: RunnableConfig, writer: StreamWriter) -> dict[str, Any]:
    """
    Identifies the user's problem based on their query.
    """
    user_id = config["configurable"]["user_id"]
    user_query = state["user_query"]

    llm = await get_user_fast_llm(state["db_session"], user_id)
    if not llm:
        return {"identified_problem": "Could not identify problem: LLM not configured."}

    prompt = get_problem_identification_prompt().format(user_query=user_query)
    messages = [
        SystemMessage(content=get_alison_system_prompt()),
        HumanMessage(content=prompt),
    ]

    response = await llm.ainvoke(messages)
    identified_problem = response.content.strip()

    return {"identified_problem": identified_problem}

async def search_knowledge_base(state: AlisonState, config: RunnableConfig, writer: StreamWriter) -> dict[str, Any]:
    """
    Searches the knowledge base for troubleshooting guides related to the identified problem.
    """
    identified_problem = state["identified_problem"]
    if not identified_problem:
        return {"troubleshooting_steps": [], "visual_aids": []}

    # Construct path to the knowledge base file relative to this file's location
    current_dir = pathlib.Path(__file__).parent
    knowledge_base_path = current_dir.parent / "alison_docs" / "classroom_faq.json"

    try:
        with open(knowledge_base_path, "r") as f:
            faq_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"troubleshooting_steps": [], "visual_aids": []}

    keywords = identified_problem.lower().split()

    # Simple keyword search in the JSON data
    for item in faq_data:
        question = item.get("question", "").lower()
        if any(keyword in question for keyword in keywords):
            # Return the answer as the troubleshooting step
            return {"troubleshooting_steps": [item.get("answer", "")], "visual_aids": []}

    return {"troubleshooting_steps": [], "visual_aids": []}

async def generate_troubleshooting_response(state: AlisonState, config: RunnableConfig, writer: StreamWriter) -> dict[str, Any]:
    """
    Generates a response with troubleshooting steps.
    """
    troubleshooting_steps = state.get("troubleshooting_steps")
    if not troubleshooting_steps:
        return {"escalation_required": True}

    # The response is now the answer from the FAQ
    final_response = troubleshooting_steps[0]
    return {"final_response": final_response}

async def handle_escalation(state: AlisonState, config: RunnableConfig, writer: StreamWriter) -> dict[str, Any]:
    """
    Generates a response for escalating the issue to IT support.
    """
    user_id = config["configurable"]["user_id"]
    identified_problem = state["identified_problem"]

    llm = await get_user_fast_llm(state["db_session"], user_id)
    if not llm:
        return {"final_response": "I am unable to resolve this issue. Please contact IT support."}

    prompt = get_escalation_prompt().format(identified_problem=identified_problem)
    messages = [
        SystemMessage(content=get_alison_system_prompt()),
        HumanMessage(content=prompt),
    ]

    response = await llm.ainvoke(messages)
    escalation_message = response.content.strip()

    return {"final_response": escalation_message}
