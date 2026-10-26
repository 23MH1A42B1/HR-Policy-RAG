"""Comprehensive test suite for Authentication, Authorization, and Production GCS Storage.

Tests:
1. Authentication state (UserInfo models, Google OAuth URL construction, token validation).
2. Authorization (email domain allow-list, user allow-list, admin allow-list).
3. Employee access vs Admin access (RBAC enforcement).
4. Unauthorized request deflection ("Your account is not authorized to access the HR Policy Assistant.").
5. GCS production storage abstraction (upload, download, list, delete, category hierarchy).
6. End-to-end admin policy pipeline (GCS -> Parser -> Chunker -> Embeddings -> Qdrant -> BM25).
"""

import tempfile
from unittest import mock
import pytest

from app.auth.authorizer import (
    UNAUTHORIZED_MESSAGE,
    Authorizer,
)
from app.auth.google_auth import GoogleAuthService
from app.auth.user import UserInfo
from app.embeddings.embedder import MockEmbeddingService
from app.ingestion.admin_pipeline import AdminPolicyPipeline
from app.retrieval.bm25_search import get_bm25_index
from app.retrieval.qdrant_store import QdrantStore
from app.storage.gcs_storage import (
    GCSStorageService,
)


# =====================================================================
# 1. Authentication State Tests
# =====================================================================

def test_user_info_model_defaults_and_admin_sync():
    """Verify UserInfo model defaults and role/admin synchronization."""
    employee = UserInfo(
        email="john@company.com",
        name="John Doe",
        user_id="usr_john",
        role="employee",
    )
    assert employee.role == "employee"
    assert employee.is_admin is False

    admin = UserInfo(
        email="admin@company.com",
        name="Admin User",
        user_id="usr_admin",
        role="admin",
    )
    assert admin.role == "admin"
    assert admin.is_admin is True


def test_google_auth_service_unconfigured_detection():
    """Verify GoogleAuthService detects missing client credentials cleanly."""
    svc = GoogleAuthService(client_id=None, client_secret=None)
    assert svc.is_oauth_configured() is False
    with pytest.raises(ValueError, match="Google Client ID is not configured"):
        svc.get_authorization_url()


def test_google_auth_service_authorization_url_generation():
    """Verify GoogleAuthService constructs valid standard Google OAuth2 URL."""
    svc = GoogleAuthService(
        client_id="test-client-id.apps.googleusercontent.com",
        client_secret="test-secret",
        redirect_uri="http://localhost:8501",
    )
    assert svc.is_oauth_configured() is True
    auth_url = svc.get_authorization_url(state="test_state_123")
    assert "https://accounts.google.com/o/oauth2/v2/auth" in auth_url
    assert "client_id=test-client-id.apps.googleusercontent.com" in auth_url
    assert "redirect_uri=http%3A%2F%2Flocalhost%3A8501" in auth_url
    assert "state=test_state_123" in auth_url
    assert "scope=openid+email+profile" in auth_url or "scope=openid%20email%20profile" in auth_url


def test_google_auth_mock_authentication():
    """Verify simulated authentication returns valid UserInfo for authorized users."""
    authorizer = Authorizer(
        allowed_domains=["company.com"],
        allowed_users=["contractor@external.org"],
        admin_users=["admin@company.com"],
    )
    svc = GoogleAuthService(authorizer=authorizer)

    user = svc.authenticate_mock(email="jane.doe@company.com", name="Jane Doe")
    assert user.email == "jane.doe@company.com"
    assert user.name == "Jane Doe"
    assert user.role == "employee"
    assert user.is_admin is False


# =====================================================================
# 2. Authorization & Allow-List Tests
# =====================================================================

def test_authorization_allowed_email_domain():
    """Verify emails matching ALLOWED_EMAIL_DOMAINS are authorized as employees."""
    authorizer = Authorizer(
        allowed_domains=["company.com", "subsidiary.io"],
        allowed_users=[],
        admin_users=["admin@company.com"],
    )

    res = authorizer.check_authorization("employee@company.com")
    assert res.is_authorized is True
    assert res.role == "employee"
    assert res.is_admin is False

    res_sub = authorizer.check_authorization("engineer@subsidiary.io")
    assert res_sub.is_authorized is True
    assert res_sub.role == "employee"


def test_authorization_allowed_individual_user():
    """Verify specific individual emails in ALLOWED_USERS are granted access."""
    authorizer = Authorizer(
        allowed_domains=["company.com"],
        allowed_users=["special.contractor@partner.org", "auditor@external.com"],
        admin_users=[],
    )

    res = authorizer.check_authorization("special.contractor@partner.org")
    assert res.is_authorized is True
    assert res.role == "employee"


def test_authorization_admin_users():
    """Verify emails in ADMIN_USERS are granted administrative permissions."""
    authorizer = Authorizer(
        allowed_domains=["company.com"],
        allowed_users=[],
        admin_users=["admin@company.com", "hr-director@company.com"],
    )

    res = authorizer.check_authorization("admin@company.com")
    assert res.is_authorized is True
    assert res.role == "admin"
    assert res.is_admin is True


def test_authorization_unauthorized_requests_rejected():
    """Verify unauthorized emails are rejected with exact required message."""
    authorizer = Authorizer(
        allowed_domains=["company.com"],
        allowed_users=["contractor@partner.org"],
        admin_users=["admin@company.com"],
    )

    unauthorized_emails = [
        "intruder@attacker.com",
        "random.user@gmail.com",
        "employee@wrongcompany.org",
        "invalid-email",
        "",
    ]

    for email in unauthorized_emails:
        res = authorizer.check_authorization(email)
        assert res.is_authorized is False
        assert res.rejection_reason == UNAUTHORIZED_MESSAGE


def test_authorizer_authorize_user_raises_on_unauthorized():
    """Verify authorizer.authorize_user raises PermissionError on unauthorized users."""
    authorizer = Authorizer(allowed_domains=["company.com"], allowed_users=[], admin_users=[])
    unauthorized = UserInfo(
        email="outsider@unknown.com",
        name="Outsider",
        user_id="usr_out",
    )

    with pytest.raises(PermissionError, match=UNAUTHORIZED_MESSAGE):
        authorizer.authorize_user(unauthorized)


# =====================================================================
# 3. Employee Access vs Admin Access (RBAC) Tests
# =====================================================================

def test_rbac_admin_action_permission_check():
    """Verify employees cannot perform admin operations while admins can."""
    authorizer = Authorizer()

    employee = UserInfo(
        email="emp@company.com",
        name="Regular Employee",
        user_id="usr_emp",
        role="employee",
        is_admin=False,
    )
    admin = UserInfo(
        email="admin@company.com",
        name="HR Administrator",
        user_id="usr_adm",
        role="admin",
        is_admin=True,
    )

    assert authorizer.can_perform_admin_action(employee) is False
    assert authorizer.can_perform_admin_action(admin) is True


# =====================================================================
# 4. Production GCS Storage Abstraction Tests
# =====================================================================

def test_gcs_storage_upload_download_lifecycle():
    """Verify uploading to GCS and downloading exact bytes."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        storage = GCSStorageService(
            bucket_name="company-hr-policies",
            local_root=tmp_dir,
        )

        test_bytes = b"%PDF-1.4 Mock Leave Policy Document Content for Testing"
        dest_path = "policies/leave/annual_leave_2026.pdf"

        gcs_uri = storage.upload(test_bytes, dest_path)
        assert gcs_uri == "gs://company-hr-policies/policies/leave/annual_leave_2026.pdf"

        downloaded = storage.download(dest_path)
        assert downloaded == test_bytes

        # Verify downloading using gs:// URI format also succeeds
        downloaded_via_uri = storage.download("gs://company-hr-policies/policies/leave/annual_leave_2026.pdf")
        assert downloaded_via_uri == test_bytes


def test_gcs_storage_list_filtering_and_categories():
    """Verify listing stored policies across different category folders."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        storage = GCSStorageService(
            bucket_name="company-hr-policies",
            local_root=tmp_dir,
        )

        storage.upload(b"leave content", "policies/leave/leave.pdf")
        storage.upload(b"remote content", "policies/remote-work/remote.pdf")
        storage.upload(b"travel content", "policies/travel/travel.docx")

        all_policies = storage.list("policies/")
        assert len(all_policies) == 3
        filenames = {p.filename for p in all_policies}
        assert filenames == {"leave.pdf", "remote.pdf", "travel.docx"}

        # Filter by specific category
        leave_policies = storage.list("policies/leave/")
        assert len(leave_policies) == 1
        assert leave_policies[0].filename == "leave.pdf"
        assert leave_policies[0].category == "leave"


def test_gcs_storage_delete():
    """Verify document deletion from GCS storage."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        storage = GCSStorageService(local_root=tmp_dir)
        path = "policies/expenses/expense_guidelines.pdf"
        storage.upload(b"expense data", path)

        assert storage.exists(path) is True
        deleted = storage.delete(path)
        assert deleted is True
        assert storage.exists(path) is False

        with pytest.raises(FileNotFoundError):
            storage.download(path)


def test_gcs_storage_transparent_status_reporting():
    """Verify GCS storage reports local status transparently when cloud credentials are unset."""
    with mock.patch.dict("os.environ", {"GOOGLE_APPLICATION_CREDENTIALS": "", "GCP_ACCESS_TOKEN": ""}, clear=True):
        storage = GCSStorageService(project=None)
        assert storage.is_cloud is False
        assert "local_storage" in storage.integration_status


# =====================================================================
# 5. End-to-End Admin Policy Pipeline Tests (GCS -> Parser -> Chunker -> Embeddings -> Qdrant -> BM25)
# =====================================================================

def test_admin_pipeline_upload_index_and_search_lifecycle():
    """Verify full admin pipeline: upload to GCS, parse, chunk, embed, upsert to Qdrant and index in BM25."""
    with tempfile.TemporaryDirectory() as tmp_storage_dir:
        gcs_storage = GCSStorageService(local_root=tmp_storage_dir)
        qdrant_store = QdrantStore(url=":memory:", collection_name="test_admin_policies")
        embed_service = MockEmbeddingService(dimension=64)

        pipeline = AdminPolicyPipeline(
            gcs_storage=gcs_storage,
            qdrant_store=qdrant_store,
            embedding_service=embed_service,
        )

        # Create a valid test PDF document in memory
        from reportlab.pdfgen import canvas
        import io
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer)
        c.drawString(100, 770, "Policy Code HR-POL-2026-LEAVE")
        c.drawString(100, 750, "1. Annual Leave Entitlement")
        c.drawString(100, 730, "All full-time employees receive 25 business days of paid vacation per calendar year.")
        c.save()
        pdf_bytes = pdf_buffer.getvalue()

        # Execute upload and index
        result = pipeline.upload_and_index_policy(
            file_bytes=pdf_bytes,
            filename="HR-POL-2026-LEAVE.pdf",
            category="leave",
            version="2026",
        )

        assert result["status"] == "success"
        assert result["document"] == "HR-POL-2026-LEAVE.pdf"
        assert result["category"] == "leave"
        assert result["chunks_indexed"] > 0
        assert gcs_storage.exists("policies/leave/HR-POL-2026-LEAVE.pdf") is True

        # Verify Qdrant dense vector search returns the uploaded document
        q_vec = embed_service.embed_query("annual leave vacation entitlement")
        qdrant_results = qdrant_store.search_dense(q_vec, top_k=5)
        assert len(qdrant_results) > 0
        assert qdrant_results[0].document == "HR-POL-2026-LEAVE.pdf"

        # Verify BM25 lexical search returns the uploaded document
        bm25_results = get_bm25_index().search("HR-POL-2026-LEAVE", top_k=5)
        assert any(r.document == "HR-POL-2026-LEAVE.pdf" for r in bm25_results)

        # Verify Ingestion Status summary
        status = pipeline.get_ingestion_status()
        assert status["total_documents_gcs"] == 1
        assert status["category_breakdown"].get("leave") == 1

        # Delete policy
        del_res = pipeline.delete_policy("policies/leave/HR-POL-2026-LEAVE.pdf")
        assert del_res["status"] == "deleted"
        assert gcs_storage.exists("policies/leave/HR-POL-2026-LEAVE.pdf") is False
