import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from surfsense_backend.app.agents.alison.nodes import (
    identify_problem,
    search_knowledge_base,
    generate_troubleshooting_response,
    handle_escalation,
)
from surfsense_backend.app.agents.alison.state import AlisonState
from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamWriter

@pytest.fixture
def initial_state():
    return AlisonState(
        user_query="My projector is not working.",
        identified_problem=None,
        troubleshooting_steps=None,
        visual_aids=None,
        escalation_required=False,
        user_role="teacher",
        final_response=None,
        db_session=MagicMock(),
        chat_history=[],
        streaming_service=MagicMock(),
    )

@pytest.fixture
def config():
    return RunnableConfig(configurable={"user_id": "123", "user_role": "teacher"})

@pytest.fixture
def writer():
    return MagicMock(spec=StreamWriter)

@pytest.mark.asyncio
@patch("surfsense_backend.app.agents.alison.nodes.get_user_fast_llm")
async def test_identify_problem(mock_get_llm, initial_state, config, writer):
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value.content = "projector not working"
    mock_get_llm.return_value = mock_llm

    result = await identify_problem(initial_state, config, writer)
    assert "identified_problem" in result
    assert result["identified_problem"] == "projector not working"

@pytest.mark.asyncio
@patch("surfsense_backend.app.agents.alison.nodes.pathlib.Path.open")
@patch("surfsense_backend.app.agents.alison.nodes.json.load")
async def test_search_knowledge_base(mock_json_load, mock_path_open, initial_state, config, writer):
    initial_state["identified_problem"] = "projector not turning on"
    mock_json_load.return_value = [
        {"question": "projector not turning on", "answer": "check power cable"}
    ]

    result = await search_knowledge_base(initial_state, config, writer)
    assert "troubleshooting_steps" in result
    assert result["troubleshooting_steps"] == ["check power cable"]

@pytest.mark.asyncio
async def test_generate_troubleshooting_response(initial_state, config, writer):
    initial_state["troubleshooting_steps"] = ["check power cable"]
    result = await generate_troubleshooting_response(initial_state, config, writer)
    assert "final_response" in result
    assert result["final_response"] == "check power cable"

@pytest.mark.asyncio
async def test_generate_troubleshooting_response_escalation(initial_state, config, writer):
    initial_state["troubleshooting_steps"] = []
    result = await generate_troubleshooting_response(initial_state, config, writer)
    assert "escalation_required" in result
    assert result["escalation_required"] is True

@pytest.mark.asyncio
@patch("surfsense_backend.app.agents.alison.nodes.get_user_fast_llm")
async def test_handle_escalation(mock_get_llm, initial_state, config, writer):
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value.content = "escalating to IT support"
    mock_get_llm.return_value = mock_llm
    initial_state["identified_problem"] = "projector on fire"

    result = await handle_escalation(initial_state, config, writer)
    assert "final_response" in result
    assert result["final_response"] == "escalating to IT support"
