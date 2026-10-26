"""Tests for redesigned Authentication flow, Sidebar, Citations, and Role-Based UI."""

from unittest import mock
import pytest

from app.auth.authorizer import UNAUTHORIZED_MESSAGE, Authorizer
from app.auth.google_auth import get_authenticated_user, is_authenticated
from app.auth.user import UserInfo
from app.ui.components import Citation, format_citation


def test_format_citation_excludes_internal_retrieval_metadata():
    """Verify format_citation strips internal scores, chunk IDs, and vectors for employees."""
    raw_source = {
        "document_name": "02_XYZ_Leave_Policy.pdf",
        "policy_id": "HR-LEAVE-001",
        "version": "2.0",
        "section": "1. Active Entitlements",
        "page": 1,
        "snippet": "Employees receive 18 working days of Annual Leave.",
        # Internal debug fields that MUST NOT leak
        "chunk_id": "leave_chunk_001_secret",
        "score": 0.8923,
        "dense_rank": 1,
        "bm25_rank": 2,
        "embedding": [0.12, 0.45, -0.67],
        "filepath": "/internal/servers/data/raw/02_XYZ_Leave_Policy.pdf",
    }

    clean = format_citation(raw_source)

    # Required employee-facing fields
    assert clean["document_name"] == "Leave Policy"
    assert clean["policy_id"] == "HR-LEAVE-001"
    assert clean["version"] == "2.0"
    assert clean["section"] == "1. Active Entitlements"
    assert clean["page"] == 1
    assert "18 working days" in clean["snippet"]

    # Internal fields must NOT be present
    assert "chunk_id" not in clean
    assert "score" not in clean
    assert "dense_rank" not in clean
    assert "bm25_rank" not in clean
    assert "embedding" not in clean
    assert "filepath" not in clean


def test_format_citation_with_pydantic_citation():
    """Verify format_citation handles Citation model instances."""
    cit = Citation(
        source="Remote Work Policy",
        document_name="Remote Work Policy",
        policy_id="HR-WFH-001",
        version="1.0",
        page=1,
        section="Core Hours",
        snippet="Core hours are 10 AM to 5 PM IST.",
    )

    clean = format_citation(cit)
    assert clean["document_name"] == "Remote Work Policy"
    assert clean["policy_id"] == "HR-WFH-001"
    assert clean["version"] == "1.0"
    assert clean["section"] == "Core Hours"
    assert clean["page"] == 1


def test_authorizer_rejects_unauthorized_with_exact_message():
    """Verify unauthorized email generates exact required message."""
    authorizer = Authorizer(
        allowed_domains=["xyz.com"],
        allowed_users=["contractor@external.com"],
        admin_users=["admin@xyz.com"],
    )

    res = authorizer.check_authorization("stranger@gmail.com")
    assert res.is_authorized is False
    assert res.rejection_reason == UNAUTHORIZED_MESSAGE
    assert "Your account is authenticated, but you are not authorized" in res.rejection_reason


def test_authorizer_assigns_correct_roles():
    """Verify role assignment between admin and employee."""
    authorizer = Authorizer(
        allowed_domains=["xyz.com"],
        allowed_users=["employee@partner.org"],
        admin_users=["admin@xyz.com", "murali@xyz.com"],
    )

    # Admin
    admin_res = authorizer.check_authorization("murali@xyz.com")
    assert admin_res.is_authorized is True
    assert admin_res.role == "admin"
    assert admin_res.is_admin is True

    # Employee via domain
    emp_res = authorizer.check_authorization("staff@xyz.com")
    assert emp_res.is_authorized is True
    assert emp_res.role == "employee"
    assert emp_res.is_admin is False

    # Employee via individual user
    ext_res = authorizer.check_authorization("employee@partner.org")
    assert ext_res.is_authorized is True
    assert ext_res.role == "employee"
    assert ext_res.is_admin is False


def test_role_based_permissions():
    """Verify employee cannot perform admin actions while admin can."""
    authorizer = Authorizer()

    employee = UserInfo(
        email="regular@xyz.com",
        name="Regular Employee",
        user_id="usr_01",
        role="employee",
    )
    assert authorizer.can_perform_admin_action(employee) is False

    admin = UserInfo(
        email="murali@xyz.com",
        name="Murali Nadipena",
        user_id="usr_02",
        role="admin",
    )
    assert authorizer.can_perform_admin_action(admin) is True


def test_feedback_recording_and_metrics(tmp_path, monkeypatch):
    """Verify recording feedback updates log and metrics correctly."""
    from app.ui.feedback import FEEDBACK_FILE, generate_followup_suggestions, get_feedback_metrics, record_feedback

    test_file = tmp_path / "test_feedback.json"
    monkeypatch.setattr("app.ui.feedback.FEEDBACK_FILE", test_file)

    # Initial state
    initial_metrics = get_feedback_metrics()
    assert initial_metrics["total_feedback"] == 0
    assert initial_metrics["satisfaction_rate"] == 100.0

    # Record helpful vote
    ok1 = record_feedback(
        message_id="msg_1",
        query="How many annual leave days do I have?",
        answer="You have 18 annual leave days.",
        rating="helpful",
        user_email="employee@xyz.com",
    )
    assert ok1 is True

    # Record unhelpful vote
    ok2 = record_feedback(
        message_id="msg_2",
        query="What is the travel per diem?",
        answer="I do not have specific per diem rates.",
        rating="unhelpful",
        user_email="contractor@xyz.com",
    )
    assert ok2 is True

    # Metrics computation
    metrics = get_feedback_metrics()
    assert metrics["total_feedback"] == 2
    assert metrics["helpful_count"] == 1
    assert metrics["unhelpful_count"] == 1
    assert metrics["satisfaction_rate"] == 50.0

    # Changing vote for msg_2 to helpful
    ok3 = record_feedback(
        message_id="msg_2",
        query="What is the travel per diem?",
        answer="Updated answer.",
        rating="helpful",
        user_email="contractor@xyz.com",
    )
    assert ok3 is True
    updated_metrics = get_feedback_metrics()
    assert updated_metrics["total_feedback"] == 2
    assert updated_metrics["helpful_count"] == 2
    assert updated_metrics["unhelpful_count"] == 0
    assert updated_metrics["satisfaction_rate"] == 100.0


def test_generate_followup_suggestions():
    """Verify follow-up suggestion chips match topic queries."""
    from app.ui.feedback import generate_followup_suggestions

    leave_suggs = generate_followup_suggestions(
        query="How to apply for leave?",
        answer="Apply via HR portal.",
    )
    assert len(leave_suggs) >= 2
    assert any("leave" in s.lower() for s in leave_suggs)

    wfh_suggs = generate_followup_suggestions(
        query="Can I work from home?",
        answer="Hybrid policy allows up to 2 days.",
    )
    assert len(wfh_suggs) >= 2
    assert any("office" in s.lower() or "hours" in s.lower() for s in wfh_suggs)


def test_multilingual_engine_detection_and_directives():
    """Verify script detection and prompt directives for Hindi, Telugu, and Tamil."""
    from app.rag.multilingual import (
        STARTER_QUESTIONS,
        WELCOME_MESSAGES,
        detect_language,
        get_language_directive,
        translate_query_to_english_for_search,
    )

    # 1. Script Detection
    assert detect_language("How many leave days?") is None
    assert detect_language("छुट्टी के नियम क्या हैं?") == "Hindi"
    assert detect_language("నాకు ఎన్ని సెలవులు ఉన్నాయి?") == "Telugu"
    assert detect_language("விடுப்பு விதிகள் என்ன?") == "Tamil"

    # 2. Query Enhancement for English RAG Search
    telugu_enhanced = translate_query_to_english_for_search("నాకు సెలవు కావాలి", active_language="Telugu")
    assert "leave" in telugu_enhanced

    hindi_enhanced = translate_query_to_english_for_search("छुट्टी की नीति", active_language="Hindi")
    assert "leave" in hindi_enhanced

    tamil_enhanced = translate_query_to_english_for_search("விடுப்பு விவரங்கள்", active_language="Tamil")
    assert "leave" in tamil_enhanced

    # 3. Directives
    assert "HINDI" in get_language_directive("Hindi")
    assert "TELUGU" in get_language_directive("Telugu")
    assert "TAMIL" in get_language_directive("Tamil")
    assert get_language_directive("English") == ""

    # 4. Starters & Welcome
    for lang in ["English", "Hindi", "Telugu", "Tamil"]:
        assert lang in STARTER_QUESTIONS
        assert len(STARTER_QUESTIONS[lang]) == 4
        assert lang in WELCOME_MESSAGES


def test_multilingual_conversational_greetings():
    """Verify conversational greetings respond in Telugu, Hindi, and Tamil."""
    from app.agent.scope_checker import get_scope_checker

    checker = get_scope_checker()

    # Telugu
    reply_te = checker.check_conversational_intent("నమస్కారం")
    assert reply_te is not None
    assert "HR పాలసీ" in reply_te

    # Hindi
    reply_hi = checker.check_conversational_intent("नमस्ते")
    assert reply_hi is not None
    assert "HR पॉलिसी" in reply_hi

    # Tamil
    reply_ta = checker.check_conversational_intent("வணக்கம்")
    assert reply_ta is not None
    assert "HR கொள்கை" in reply_ta

