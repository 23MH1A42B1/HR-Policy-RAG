"""Tests for UI components and data models."""

from app.ui.components import Citation
from app.ui.chat import INITIAL_ASSISTANT_MESSAGE, handle_user_query


def test_citation_model_validation():
    """Verify Citation data model parses valid attributes properly."""
    citation = Citation(
        source="Leave Policy",
        page=4,
        section="Annual Leave",
        snippet="Employees accrue 20 days of paid annual leave per calendar year.",
    )
    assert citation.source == "Leave Policy"
    assert citation.page == 4
    assert citation.section == "Annual Leave"
    assert "20 days" in citation.snippet


def test_citation_model_optional_fields():
    """Verify Citation works with minimal required fields."""
    citation = Citation(source="Workplace Safety Guidelines")
    assert citation.source == "Workplace Safety Guidelines"
    assert citation.page is None
    assert citation.section is None
    assert citation.snippet is None


def test_initial_assistant_message_content():
    """Verify welcome message matches specified requirements."""
    assert INITIAL_ASSISTANT_MESSAGE["role"] == "assistant"
    assert (
        INITIAL_ASSISTANT_MESSAGE["content"]
        == "Hello! I'm your HR Policy Assistant. Ask me anything about company HR policies."
    )


def test_query_response_structure():
    """Verify handle_user_query returns a valid assistant response with structured content."""
    query = "How many annual leave days do employees get?"
    response = handle_user_query(query)

    assert response["role"] == "assistant"
    assert isinstance(response["content"], str)
    assert len(response["content"]) > 10
    assert isinstance(response["citations"], list)
    assert "retrieval_results" in response
