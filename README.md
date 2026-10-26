# HR Policy RAG Assistant

An enterprise-grade, observable, and secured AI retrieval-augmented generation (RAG) assistant designed for employees and People Operations teams to query authoritative company HR policies (leave, remote work, expenses, attendance, and code of conduct).

---

## 📌 Architecture Overview

```
                                  [ Employee / HR Admin ]
                                             │ (HTTPS)
                                             ▼
                               ┌───────────────────────────┐
                               │  Google Cloud Run Service │
                               │  (Streamlit UI + Python)  │
                               └─────────────┬─────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       │ Google OAuth 2.0 / OIDC Authentication    │
                       │ Role-Based Access Control (Employee/Admin)│
                       └─────────────────────┬─────────────────────┘
                                             │
                        ┌────────────────────▼────────────────────┐
                        │ Security Layer (Stage 1: Input Guard)   │
                        │ - Prompt Injection Detection            │
                        │ - Unicode Normalization & Token Flooding│
                        │ - Google Model Armor Input Screening    │
                        └────────────────────┬────────────────────┘
                                             │
                        ┌────────────────────▼────────────────────┐
                        │ Semantic Cache (Cosine Similarity)      │
                        │ - Checks Cached Fresh Embeddings        │
                        │ - Automatic Version Invalidation        │
                        └────────────────────┬────────────────────┘
                                             │ (If Cache Miss)
                     ┌───────────────────────┴───────────────────────┐
                     ▼                                               ▼
         ┌───────────────────────┐                       ┌───────────────────────┐
         │ Qdrant Vector Store   │                       │ BM25 Lexical Index    │
         │ (FastEmbed 384-dim)   │                       │ (Exact IDs / Terms)   │
         └───────────┬───────────┘                       └───────────┬───────────┘
                     │                                               │
                     └───────────────────────┬───────────────────────┘
                                             │
                               ┌─────────────▼─────────────┐
                               │ Result Fusion (RRF / WLS) │
                               │ Top-20 Candidate Pool     │
                               └─────────────┬─────────────┘
                                             │
                               ┌─────────────▼─────────────┐
                               │ Jina Cross-Encoder Rerank │
                               │ Top-5 High-Precision Chunks│
                               └─────────────┬─────────────┘
                                             │
                        ┌────────────────────▼────────────────────┐
                        │ Untrusted Context Sanitizer             │
                        │ (Neutralizes In-Document Injections)    │
                        └────────────────────┬────────────────────┘
                                             │
                               ┌─────────────▼─────────────┐
                               │ LiteLLM Resilient Router  │
                               │ Primary: Vertex AI Gemini │
                               │ Failover: Groq Llama 3.3  │
                               └─────────────┬─────────────┘
                                             │
                        ┌────────────────────▼────────────────────┐
                        │ Security Layer (Stage 3: Output Guard)  │
                        │ - Credential & API Key Redaction        │
                        │ - System Prompt Shielding               │
                        │ - Model Armor Output Screening          │
                        └────────────────────┬────────────────────┘
                                             │
                               ┌─────────────▼─────────────┐
                               │ Verified Citations & UI   │
                               │ Response Display          │
                               └─────────────┬─────────────┘
                                             │ (Telemetry)
                               ┌─────────────▼─────────────┐
                               │ LangSmith Tracing & Obs.  │
                               │ Latency, Scores, Tokens   │
                               └───────────────────────────┘
```

---

## 📂 Project Structure

```text
hr-policy-rag/
├── .dockerignore                 # Excludes secrets, caches, and virtual environments
├── .env.example                  # Template environment configuration (no secrets)
├── .gitignore                    # Git tracking rules
├── .streamlit/
│   └── config.toml               # Streamlit theme & server configuration
├── Dockerfile                    # Multi-stage production container image
├── pyproject.toml                # Package metadata, dependencies, & pytest config
├── README.md                     # Comprehensive deployment & architecture guide
├── uv.lock                       # Deterministic dependency lockfile
│
├── app/
│   ├── __init__.py
│   ├── config.py                 # Pydantic BaseSettings & configuration
│   ├── main.py                   # Streamlit application entrypoint & routing
│   │
│   ├── agent/                    # Autonomous agent orchestration
│   │   ├── agent.py              # Guarded HR policy agent with observability
│   │   ├── memory.py             # Multi-turn conversational memory & follow-ups
│   │   ├── scope_checker.py      # HR domain boundary guardrail
│   │   └── tools.py              # Controlled policy search tool
│   │
│   ├── auth/                     # Authentication & Access Control
│   │   ├── authorizer.py         # Domain & user allow-lists (RBAC)
│   │   ├── google_auth.py        # Google OAuth 2.0 / OIDC verification
│   │   └── user.py               # UserInfo & UserRole schemas
│   │
│   ├── cache/                    # Semantic Caching
│   │   └── semantic_cache.py     # Vector cache with version invalidation
│   │
│   ├── embeddings/               # Dense Vector Embeddings
│   │   └── embedder.py           # FastEmbed ONNX BAAI/bge-small-en-v1.5
│   │
│   ├── evaluation/               # Quantitative Evaluation
│   │   └── metrics.py            # 7 RAG metrics calculator & summary statistics
│   │
│   ├── ingestion/                # Document Parsing & Chunking
│   │   ├── admin_pipeline.py     # Live re-indexing pipeline for admin uploads
│   │   ├── chunker.py            # Section-aware recursive text chunker
│   │   ├── cleaner.py            # Normalization & whitespace cleaner
│   │   ├── docx_loader.py        # Word document extractor
│   │   ├── loader.py             # Unified document loader dispatcher
│   │   ├── metadata.py           # DocumentChunk & metadata models
│   │   ├── pdf_loader.py         # PDF page-by-page extractor
│   │   └── pptx_loader.py        # PowerPoint slide extractor
│   │
│   ├── llm/                      # Model Routing Layer
│   │   ├── groq.py               # Groq Llama 3.3 integration
│   │   ├── litellm_router.py     # Resilient router with automatic failover
│   │   └── vertex.py             # Google Cloud Vertex AI Gemini integration
│   │
│   ├── observability/            # Telemetry & Tracing
│   │   └── langsmith_tracer.py   # LangSmith tracing with transparent local fallback
│   │
│   ├── rag/                      # RAG Generation & Prompts
│   │   ├── citations.py          # Grounded citation extractor & formatter
│   │   ├── generator.py          # LLM service abstraction
│   │   ├── pipeline.py           # Baseline RAG query pipeline
│   │   └── prompt.py             # Strict system & context prompt builder
│   │
│   ├── retrieval/                # Hybrid Search & Reranking
│   │   ├── bm25_search.py        # Lexical BM25 search index
│   │   ├── hybrid_search.py      # Reciprocal Rank Fusion & Weighted Score Fusion
│   │   ├── qdrant_store.py       # Qdrant vector database client
│   │   └── reranker.py           # Jina cross-encoder reranker
│   │
│   ├── security/                 # Multi-Tier Guardrails
│   │   ├── input_guard.py        # Length, Unicode, & flood screening
│   │   ├── model_armor.py        # Google Model Armor client & fallback
│   │   ├── output_guard.py       # Secret redaction & hallucination mitigation
│   │   └── prompt_injection.py   # Direct & indirect injection detector
│   │
│   ├── storage/                  # Production Document Storage
│   │   └── gcs_storage.py        # Google Cloud Storage integration
│   │
│   ├── ui/                       # Frontend Presentation
│   │   ├── admin.py              # Policy management & upload console
│   │   ├── chat.py               # Interactive chat state
│   │   ├── components.py         # Visual components & cards
│   │   ├── login.py              # Corporate login gate
│   │   └── sidebar.py            # Navigation & session diagnostics
│   │
│   └── utils/
│       └── logging.py            # Structured sanitizing logger
│
├── data/
│   ├── evaluation/
│   │   ├── latest_results.json   # Persistent benchmark run outputs
│   │   └── questions.json        # 15 ground-truth evaluation questions
│   ├── processed/                # Extracted chunk JSON & manifest data
│   └── raw/                      # Authoritative sample policy documents
│
├── scripts/
│   ├── ask_policy.py             # Interactive CLI Q&A tool
│   ├── check_env.py              # Environment configuration validator
│   ├── create_qdrant_collection.py # Vector collection initializer
│   ├── evaluate.py               # Quantitative benchmark runner
│   ├── generate_sample_policies.py # Test policy generator
│   ├── healthcheck.py            # Production diagnostic health check
│   ├── ingest_documents.py       # Batch document ingestion script
│   └── search_dense.py           # Dense vector search test CLI
│
└── tests/                        # 111 Unit, Integration, & Security Tests
    ├── test_agent.py
    ├── test_app.py
    ├── test_auth_storage.py
    ├── test_chunking.py
    ├── test_components.py
    ├── test_config.py
    ├── test_embedder.py
    ├── test_evaluation.py
    ├── test_hybrid.py
    ├── test_ingestion.py
    ├── test_llm_router.py
    ├── test_qdrant.py
    ├── test_rag.py
    └── test_security.py
```

---

## 💻 Local Development

### 1. Prerequisites
- Python 3.11+
- [uv package manager](https://docs.astral.sh/uv/)

### 2. Environment Setup
Clone the repository and install dependencies using `uv`:
```bash
git clone <repository-url>
cd "HR Policy Rag"

# Create virtual environment and install all dependencies
uv sync
```

### 3. Configure Local Environment
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Fill in the necessary credentials (see [Environment Variables](#environment-variables)). For initial local development, external credentials are optional; the application falls back safely to in-memory/BM25 and mock modes.

### 4. Ingest Sample Documents
Process the raw policies in `data/raw/` into search chunks:
```bash
uv run python scripts/ingest_documents.py
```

### 5. Run Health Diagnostic
Verify all subsystems before starting:
```bash
uv run python scripts/healthcheck.py
```

### 6. Start Local Streamlit Application
```bash
uv run streamlit run app/main.py --server.port=8501
```
Open `http://localhost:8501` in your browser.

### 7. Run Test Suite & Evaluation Benchmark
```bash
# Run unit and integration tests
uv run pytest tests/ -v

# Run the 7-dimension RAG evaluation benchmark
uv run python scripts/evaluate.py
```

---

## 🐳 Docker

The application includes a minimal, multi-stage production Dockerfile (`Dockerfile`) with an unprivileged non-root user (`appuser`, UID 10001) and dynamic Cloud Run `$PORT` binding.

### 1. Build the Docker Image Locally
```bash
docker build -t hr-policy-rag .
```

### 2. Run the Container Locally
```bash
docker run -p 8080:8080 --env-file .env hr-policy-rag
```
Access the application at `http://localhost:8080`.

### 3. Container Security Highlights:
- **Base Image:** `python:3.11-slim` with `uv` copied from `ghcr.io/astral-sh/uv:0.5.24`.
- **Non-Root Execution:** Runs as `appuser:appgroup` (UID/GID 10001).
- **Zero In-Image Secrets:** `.dockerignore` blocks `.env`, `*.toml`, key files, and local logs from entering the build context.
- **Health Check Probe:** Validates HTTP 200 on `/_stcore/health` every 30 seconds.

---

## ☁️ Google Cloud Setup

Follow these steps to prepare your Google Cloud environment:

### 1. Install & Authenticate Google Cloud SDK
```bash
# Login to your Google Cloud account
gcloud auth login

# Set your project ID
gcloud config set project YOUR_PROJECT_ID
```

### 2. Enable Required Google Cloud APIs
```bash
gcloud services enable \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    secretmanager.googleapis.com \
    storage.googleapis.com \
    aiplatform.googleapis.com \
    cloudbuild.googleapis.com
```

### 3. Create Dedicated Cloud Run Service Account
```bash
gcloud iam service-accounts create hr-policy-rag-sa \
    --description="Runtime Service Account for HR Policy RAG Assistant" \
    --display-name="hr-policy-rag-sa"
```

---

## 📦 Artifact Registry

### 1. Create Artifact Registry Docker Repository
```bash
gcloud artifacts repositories create hr-policy-rag-repo \
    --repository-format=docker \
    --location=us-central1 \
    --description="Docker repository for HR Policy RAG Assistant"
```

### 2. Configure Docker Authentication
```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
```

### 3. Build & Push via Cloud Build (Recommended Serverless Method)
Building server-side avoids needing a local Docker daemon:
```bash
gcloud builds submit --tag us-central1-docker.pkg.dev/YOUR_PROJECT_ID/hr-policy-rag-repo/hr-policy-rag:latest .
```

### 4. Build & Push via Local Docker
```bash
docker tag hr-policy-rag us-central1-docker.pkg.dev/YOUR_PROJECT_ID/hr-policy-rag-repo/hr-policy-rag:latest
docker push us-central1-docker.pkg.dev/YOUR_PROJECT_ID/hr-policy-rag-repo/hr-policy-rag:latest
```

---

## 🔐 Secret Manager

Production credentials must never be baked into Docker images or committed to Git. Store them in Google Secret Manager.

### 1. Create Secrets in Secret Manager
```bash
# Vector Database
echo -n "YOUR_QDRANT_API_KEY" | gcloud secrets create qdrant-api-key --data-file=-

# Cross-Encoder Reranker
echo -n "YOUR_JINA_API_KEY" | gcloud secrets create jina-api-key --data-file=-

# Failover LLM
echo -n "YOUR_GROQ_API_KEY" | gcloud secrets create groq-api-key --data-file=-

# Observability
echo -n "YOUR_LANGSMITH_API_KEY" | gcloud secrets create langsmith-api-key --data-file=-

# Google OAuth Client Secret
echo -n "YOUR_GOOGLE_CLIENT_SECRET" | gcloud secrets create google-client-secret --data-file=-
```

### 2. Grant Secret Access to the Service Account
```bash
SERVICE_ACCOUNT_EMAIL="hr-policy-rag-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com"

for SECRET in qdrant-api-key jina-api-key groq-api-key langsmith-api-key google-client-secret; do
    gcloud secrets add-iam-policy-binding $SECRET \
        --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
        --role="roles/secretmanager.secretAccessor"
done
```

---

## 🪣 Cloud Storage (GCS)

The assistant uses Google Cloud Storage as its authoritative policy document repository.

### 1. Create the Storage Bucket
```bash
gcloud storage buckets create gs://company-hr-policies-YOUR_PROJECT_ID \
    --location=us-central1 \
    --uniform-bucket-level-access
```

### 2. Establish Policy Folder Hierarchy
```bash
# Structure policy documents into categories
touch .keep
gcloud storage cp .keep gs://company-hr-policies-YOUR_PROJECT_ID/policies/leave/.keep
gcloud storage cp .keep gs://company-hr-policies-YOUR_PROJECT_ID/policies/remote-work/.keep
gcloud storage cp .keep gs://company-hr-policies-YOUR_PROJECT_ID/policies/expenses/.keep
gcloud storage cp .keep gs://company-hr-policies-YOUR_PROJECT_ID/policies/attendance/.keep
rm .keep
```

### 3. Grant Storage Permissions
```bash
gcloud storage buckets add-iam-policy-binding gs://company-hr-policies-YOUR_PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/storage.objectAdmin"
```

---

## 🔑 OAuth 2.0 / OIDC Setup

Only authorized employees and administrators should be able to log in.

### 1. Create OAuth 2.0 Credentials in Google Cloud Console
1. Navigate to **APIs & Services** > **OAuth consent screen**.
2. Select **Internal** (recommended for company employees) or **External** with test users.
3. Configure App name (`HR Policy Assistant`) and Support email.
4. Navigate to **Credentials** > **Create Credentials** > **OAuth client ID**.
5. Select Application type: **Web application**.
6. Set Name: `hr-policy-rag-client`.

### 2. Authorized Redirect URIs
Add the following Authorized redirect URIs:
- **Local Development:** `http://localhost:8501`
- **Cloud Run Deployment:** `https://<YOUR-CLOUD-RUN-SERVICE-URL>`

### 3. Cloud Run Service Configuration
Pass the OAuth Client ID as an environment variable and Client Secret via Secret Manager.

---

## 🚀 Cloud Run Deployment

Deploy the container to Cloud Run with auto-scaling, secret references, and health probes:

```bash
gcloud run deploy hr-policy-rag \
    --image=us-central1-docker.pkg.dev/YOUR_PROJECT_ID/hr-policy-rag-repo/hr-policy-rag:latest \
    --region=us-central1 \
    --platform=managed \
    --service-account=hr-policy-rag-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com \
    --allow-unauthenticated \
    --port=8080 \
    --memory=2Gi \
    --cpu=2 \
    --min-instances=0 \
    --max-instances=10 \
    --set-env-vars="\
APP_NAME=HR Policy RAG Assistant,\
ENVIRONMENT=production,\
GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID,\
GOOGLE_CLOUD_LOCATION=us-central1,\
GCS_BUCKET_NAME=company-hr-policies-YOUR_PROJECT_ID,\
QDRANT_URL=https://your-qdrant-cluster.cloud.qdrant.io:6333,\
QDRANT_COLLECTION_NAME=hr_policies,\
GOOGLE_CLIENT_ID=YOUR_OAUTH_CLIENT_ID.apps.googleusercontent.com,\
GOOGLE_REDIRECT_URI=https://hr-policy-rag-YOUR_HASH-uc.a.run.app,\
ALLOWED_EMAIL_DOMAINS=company.com,\
ALLOWED_USERS=employee@company.com,\
ADMIN_USERS=admin@company.com,\
LANGSMITH_TRACING=true,\
LANGSMITH_PROJECT=hr-policy-rag-prod" \
    --set-secrets="\
QDRANT_API_KEY=qdrant-api-key:latest,\
JINA_API_KEY=jina-api-key:latest,\
GROQ_API_KEY=groq-api-key:latest,\
LANGSMITH_API_KEY=langsmith-api-key:latest,\
GOOGLE_CLIENT_SECRET=google-client-secret:latest"
```

---

## 📊 Observability (LangSmith)

Full lifecycle tracing captures each stage of execution:
```
User Question -> Input Security -> Semantic Cache -> Retrieval -> Fusion -> Reranking -> LLM Generation -> Output Security -> Final Answer
```

- **Metrics Tracked:** Latency per stage, retrieved document lists, dense & BM25 scores, Jina reranker scores, active LLM model, failover status, token usage statistics, and execution errors.
- **PII & Secret Sanitization:** Traces are scrubbed of API keys, tokens, and PII before transmission.
- **Local Fallback:** When `LANGSMITH_API_KEY` is omitted, the tracer functions locally without making fake cloud calls.

---

## 🔍 Vector Store (Qdrant)

- **Collection Name:** `hr_policies`
- **Vector Dimension:** 384 (`BAAI/bge-small-en-v1.5`)
- **Distance Metric:** Cosine
- **Fallback:** If Qdrant is unreachable or unconfigured, the retriever automatically continues operating using the local BM25 lexical index.

Initialize collection via script:
```bash
uv run python scripts/create_qdrant_collection.py
```

---

## 🎯 Cross-Encoder Reranking (Jina AI)

- **Model:** `jina-reranker-v2-base-multilingual`
- **Gating Pipeline:**
  - Hybrid retrieval pulls Top 20 candidate passages (Dense + BM25).
  - Jina cross-encoder computes deep semantic scores and reranks to Top 5.
  - Only Top 5 high-precision excerpts are passed into the LLM context prompt.
- **Truthful Degradation:** If `JINA_API_KEY` is not present, top hybrid candidates are passed forward with a clear degraded-mode log indicator.

---

## 🤖 LLMs: Vertex AI Gemini & Groq Failover

The routing layer provides high availability across cloud providers:

1. **Primary Route:** Google Cloud **Vertex AI Gemini** (`gemini-1.5-pro`).
   - Authenticated via the Cloud Run Service Account (`roles/aiplatform.user`).
2. **Automatic Failover:** **Groq** (`llama-3.3-70b-versatile`).
   - If Vertex AI experiences rate limits, timeouts, or quota errors, the request is instantly re-routed to Groq with zero service interruption.
   - User response transparently notes fallback utilization.

---

## 🛡️ Security Architecture

1. **Input Security Guard (`app/security/input_guard.py`):**
   - Strips invisible Unicode characters, bidirectional override overrides, and zero-width spaces.
   - Restricts maximum query length to 1,500 characters and 300 words.
2. **Prompt Injection Detection (`app/security/prompt_injection.py`):**
   - Detects direct injection attacks ("ignore instructions", "reveal system prompt", DAN mode).
3. **Untrusted Document Screening:**
   - Ingested policy files are treated as untrusted data. Neutralizes indirect prompt injections hidden in policy excerpts.
4. **Google Model Armor (`app/security/model_armor.py`):**
   - Enterprise content filtering and safety screening.
5. **Output Security Guard (`app/security/output_guard.py`):**
   - Automatically redacts leaked API keys (`gsk_...`, `AIza...`, `sk-...`, `jina_...`), database URIs, JWTs, and employee SSNs with `[REDACTED_SECRET]`.
6. **Role-Based Access Control (`app/auth/authorizer.py`):**
   - Enforces corporate domain allow-lists (`ALLOWED_EMAIL_DOMAINS`) and administrative privilege separation (`ADMIN_USERS`).

---

## 📋 Environment Variables Reference

| Variable | Required | Default | Description |
|---|:---:|:---:|---|
| `APP_NAME` | No | `HR Policy RAG Assistant` | Application UI title |
| `ENVIRONMENT` | No | `development` | Runtime environment (`development`, `staging`, `production`) |
| `PORT` | No | `8080` | HTTP port for Cloud Run / Docker |
| `GOOGLE_CLOUD_PROJECT` | Yes (in Cloud) | `""` | GCP Project ID for Vertex AI and GCS |
| `GOOGLE_CLOUD_LOCATION` | No | `us-central1` | GCP region |
| `VERTEX_MODEL` | No | `gemini-1.5-pro` | Primary Gemini model identifier |
| `GROQ_API_KEY` | Yes (for fallback)| `""` | Groq API key for LLM failover |
| `GROQ_MODEL` | No | `llama-3.3-70b-versatile` | Fallback model identifier |
| `QDRANT_URL` | Yes | `http://localhost:6333` | Qdrant vector database URL |
| `QDRANT_API_KEY` | Conditional | `""` | Qdrant API key (required for Qdrant Cloud) |
| `JINA_API_KEY` | No | `""` | Jina AI key for Cross-Encoder reranking |
| `GOOGLE_CLIENT_ID` | Yes (Auth) | `""` | Google OAuth 2.0 Web Client ID |
| `GOOGLE_CLIENT_SECRET`| Yes (Auth) | `""` | Google OAuth 2.0 Client Secret |
| `GOOGLE_REDIRECT_URI` | Yes (Auth) | `http://localhost:8501`| Authorized OAuth callback URI |
| `ALLOWED_EMAIL_DOMAINS`| Yes (Auth) | `company.com` | Comma-separated authorized email domains |
| `ADMIN_USERS` | No | `admin@company.com` | Comma-separated admin email addresses |
| `GCS_BUCKET_NAME` | Yes (Storage) | `company-hr-policies` | Google Cloud Storage bucket name |
| `LANGSMITH_TRACING` | No | `false` | Enable cloud LangSmith tracing |
| `LANGSMITH_API_KEY` | Conditional | `""` | LangSmith API key |
| `LANGSMITH_PROJECT` | No | `hr-policy-rag` | LangSmith project name |
| `SLACK_BOT_TOKEN` | Yes (for Slack) | `""` | Slack Bot User OAuth Token (`xoxb-...`) |
| `SLACK_APP_TOKEN` | Yes (for Slack) | `""` | Slack App-Level Token for Socket Mode (`xapp-...`) |
| `SLACK_ALLOWED_USERS` | No | `""` | Comma-separated allowed Slack user IDs |
| `SLACK_ADMIN_USERS` | No | `""` | Comma-separated Slack admin user IDs |
| `SLACK_RATE_LIMIT_REQUESTS` | No | `10` | Maximum Slack queries allowed per window |
| `SLACK_RATE_LIMIT_WINDOW` | No | `60` | Slack rate limit window in seconds |

---

## 💬 Slack Integration Guide

XYZ Technologies HR Policy Assistant includes a native Slack bot built on **Slack Bolt for Python** with **Socket Mode** for secure bidirectional interaction without exposing public webhooks.

> [!CAUTION]
> **Security Warning:** Never commit Slack tokens (`xoxb-...` or `xapp-...`) to GitHub or version control. Store them exclusively in your secure `.env` file or cloud secrets manager.

### Shared RAG Architecture

```
         ┌──────────────────┐               ┌──────────────────┐
         │  Streamlit Web   │               │    Slack Bot     │
         │ (chat.py UI)     │               │ (handlers.py)    │
         └────────┬─────────┘               └────────┬─────────┘
                  │                                  │
                  └─────────────────┬────────────────┘
                                    │
                         ┌──────────▼──────────┐
                         │   Shared RAG API    │
                         │ (rag_service.py)    │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │  Guarded HR Agent   │
                         │ (Scope, Armor, RAG) │
                         └─────────────────────┘
```

Both interfaces consume the exact same backend service (`answer_question`). Grounding rules, Qdrant vectors, BM25 indices, Model Armor security screening, and LLM synthesis remain completely synchronized.

---

### Step-by-Step Setup Instructions

#### 1. Create Slack App
1. Navigate to [Slack API App Management](https://api.slack.com/apps).
2. Click **Create New App** → select **From an app manifest**.
3. Choose your development workspace.
4. Copy the contents of [`slack/manifest.yml`](file:///c:/pdf/HR%20Policy%20Rag/slack/manifest.yml) and paste it into the YAML editor.
5. Click **Next** → **Create**.

#### 2. Configure Bot
- The manifest automatically assigns the display name **XYZ HR Assistant** and username `xyz_hr_assistant`.
- Verify the bot user is configured under **Features** → **App Home**.

#### 3. Configure Socket Mode
1. In the left navigation, click **Settings** → **Socket Mode**.
2. Ensure **Enable Socket Mode** is turned **ON**.

#### 4. Create App Token
1. In **Settings** → **Basic Information**, scroll to **App-Level Tokens**.
2. Click **Generate Token and Scopes**.
3. Set Token Name: `socket-mode-token`.
4. Add the `connections:write` scope.
5. Click **Generate** and copy the token starting with `xapp-...`.

#### 5. Configure OAuth Permissions
Verify the bot token scopes under **Features** → **OAuth & Permissions**:
- `app_mentions:read`
- `channels:history`
- `chat:write`
- `im:history`
- `im:read`
- `im:write`
- `commands`
- `users:read`
- `users:read.email`

#### 6. Install App to Workspace
1. Under **Settings** → **Install App**, click **Install to Workspace**.
2. Review permissions and click **Allow**.
3. Copy the **Bot User OAuth Token** starting with `xoxb-...`.

#### 7. Set Environment Variables
Add your tokens to `.env`:
```bash
SLACK_BOT_TOKEN="xoxb-your-bot-token"
SLACK_APP_TOKEN="xapp-your-app-token"
SLACK_ALLOWED_USERS="U12345,U67890"        # Optional allow-list
SLACK_ALLOWED_EMAIL_DOMAINS="xyz.com"     # Corporate domain verification
SLACK_ADMIN_USERS="U12345"
SLACK_RATE_LIMIT_REQUESTS=10
SLACK_RATE_LIMIT_WINDOW=60
```

#### 8. Run Slack Bot
Launch the Slack bot runner in Socket Mode:
```bash
uv run python scripts/run_slack.py
```
Output:
```
⚡ XYZ HR Assistant Slack Bot is running
Connected in Socket Mode. Listening for DMs, @mentions, and /ask-hr commands...
```

#### 9. Test Direct Message (DM)
1. In Slack, find **XYZ HR Assistant** under Apps.
2. Send a direct message:
   > *"How many annual leave days do I get?"*
3. Bot responds with:
   > *"You are entitled to 18 working days of Annual Leave per year."*
   > *Sources (1)*
   > 📄 Leave and Time-Off Policy (HR-LEAVE-001 • Version 2.0)

#### 10. Test @mention
In any channel where the bot is invited:
> *@XYZ HR Assistant What are the core collaboration hours?*
Bot strips the mention and responds in thread with:
> *"10:00 AM to 5:00 PM IST."*

#### 11. Test /ask-hr Slash Command
Use the command anywhere in Slack:
> `/ask-hr Can I carry forward unused annual leave days?`
The bot acknowledges immediately and responds in-channel with up to 12 days carryover rules.

---

## ⚠️ Known Limitations

1. **Document File Formats:** Supported file formats are currently PDF, DOCX, and PPTX. Legacy binary formats (`.doc`, `.ppt`) require conversion.
2. **Scanned Images:** Scanned image-only PDFs without an OCR text layer must be pre-processed with an OCR pipeline (e.g. Google Cloud Document AI) before ingestion.
3. **Session Persistence:** Streamlit in-memory state is tied to the container instance. In multi-instance auto-scaled Cloud Run environments, user sessions use browser cookies and OAuth state, while vector state is synchronized centrally via Qdrant and GCS.

