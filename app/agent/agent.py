"""Guarded HR Policy Agent orchestrating scope checking, semantic cache, memory, and retrieval.

Follows a strict execution sequence:
1. Understand the question (via conversation memory & follow-up contextualization).
2. Check scope (deflect out-of-scope queries cleanly).
3. Check semantic cache (return cached answer if fresh and matching).
4. Search HR policies using controlled retrieval tool (Dense + BM25 -> Top 20).
5. Rerank results (Jina Cross-Encoder -> Top 5).
6. Generate grounded answer (LiteLLM Router: Vertex AI Gemini primary, Groq fallback).
7. Store in cache & memory, and return verified citations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.agent.memory import MemoryManager, SessionMemory, get_memory_manager
from app.agent.scope_checker import HRScopeChecker, get_scope_checker
from app.agent.tools import PolicySearchResult, PolicySearchTool
from app.cache.semantic_cache import SemanticCache, get_semantic_cache
from app.config import get_settings
from app.embeddings.embedder import EmbeddingService, get_embedding_service
from app.observability import LangSmithService, RAGPipelineTrace, get_langsmith_service
from app.rag.citations import SourceCitation, extract_citations_from_results
from app.rag.generator import LLMConfigurationError, LLMService, get_llm_service
from app.rag.prompt import build_system_prompt, build_user_prompt
from app.security.model_armor import ModelArmorService, get_model_armor
from app.utils.logging import get_logger

logger = get_logger("app.agent.agent")


class AgentResponse(BaseModel):
    """Structured response container for the Guarded HR Policy Agent."""

    answer: str = Field(description="Generated or cached grounded response text")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Verified source citations")
    retrieval_results: List[Dict[str, Any]] = Field(default_factory=list, description="Policy excerpts passed to LLM")
    cache_hit: bool = Field(default=False, description="Whether answer was fulfilled from semantic cache")
    in_scope: bool = Field(default=True, description="Whether inquiry was within HR policy scope")
    model_used: str = Field(default="none", description="LLM model or cache identifier")
    fallback_used: bool = Field(default=False, description="Whether fallback model was activated")
    reranker_used: str = Field(default="none", description="Reranker identifier used")
    session_id: str = Field(default="", description="Conversation session ID")
    trace_id: Optional[str] = Field(default=None, description="LangSmith / local execution trace identifier")


class HRPolicyAgent:
    """Enterprise HR Policy Assistant Agent."""

    def __init__(
        self,
        search_tool: Optional[PolicySearchTool] = None,
        scope_checker: Optional[HRScopeChecker] = None,
        cache: Optional[SemanticCache] = None,
        memory_manager: Optional[MemoryManager] = None,
        model_armor: Optional[ModelArmorService] = None,
        embedder: Optional[EmbeddingService] = None,
        llm: Optional[LLMService] = None,
        tracer: Optional[LangSmithService] = None,
    ) -> None:
        self.search_tool = search_tool or PolicySearchTool()
        self.scope_checker = scope_checker or get_scope_checker()
        self.cache = cache or get_semantic_cache()
        self.memory_manager = memory_manager or get_memory_manager()
        self.model_armor = model_armor or get_model_armor()
        self.embedder = embedder or get_embedding_service()
        self.llm = llm  # If None, dynamically resolved via get_llm_service()
        self.tracer = tracer or get_langsmith_service()

    def process_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        policy_type: Optional[str] = None,
        top_k: int = 5,
        candidate_k: int = 20,
        language: str = "English",
    ) -> AgentResponse:
        """Process employee question through the guarded agent pipeline with observability tracing."""
        clean_q = query.strip()
        sid = session_id or "default_session"
        session_mem: SessionMemory = self.memory_manager.get_session(sid)

        if not clean_q:
            return AgentResponse(
                answer="Please ask a question about company HR policies.",
                session_id=sid,
            )

        logger.info(f"HR Agent received query (session={sid}, lang={language}): '{clean_q[:70]}...'")

        # Initialize Observability Trace
        trace: RAGPipelineTrace = self.tracer.start_trace(question=clean_q, session_id=sid)

        # ==============================================================
        # Security Pipeline Stage 1: Input Guard & Model Armor Screening
        # ==============================================================
        with trace.measure_step("input_security") as step_meta:
            input_screening = self.model_armor.screen_input(clean_q)
            step_meta["is_safe"] = input_screening.is_safe
            step_meta["threats_detected"] = input_screening.threats_detected

        if not input_screening.is_safe:
            logger.warning(
                f"Query rejected by Model Armor / Input Guard: threats={input_screening.threats_detected}"
            )
            rejection_text = input_screening.rejection_message or "Security check failed: input rejected."
            trace.final_answer = rejection_text
            self.tracer.finish_trace(trace)
            return AgentResponse(
                answer=rejection_text,
                in_scope=False,
                session_id=sid,
                trace_id=trace.trace_id,
            )

        sanitized_query = input_screening.sanitized_text

        # ==============================================================
        # Step 1: Understand Question & Contextualize Follow-ups
        # ==============================================================
        # Direct conversational greeting / pleasantry check
        conversational_reply = self.scope_checker.check_conversational_intent(sanitized_query, language=language)
        if conversational_reply:
            logger.info(f"Agent answered conversational greeting: '{sanitized_query}'")
            session_mem.add_turn(sanitized_query, conversational_reply)
            trace.final_answer = conversational_reply
            self.tracer.finish_trace(trace)
            return AgentResponse(
                answer=conversational_reply,
                in_scope=True,
                session_id=sid,
                trace_id=trace.trace_id,
            )

        contextualized_query = session_mem.contextualize_query(sanitized_query)

        # ==============================================================
        # Step 2: Check Scope
        # ==============================================================
        scope_assessment = self.scope_checker.check_scope(contextualized_query)
        if not scope_assessment.is_in_scope:
            logger.info(f"Agent deflected out-of-scope query: '{sanitized_query[:50]}...'")
            trace.final_answer = scope_assessment.decline_message
            self.tracer.finish_trace(trace)
            return AgentResponse(
                answer=scope_assessment.decline_message,
                in_scope=False,
                session_id=sid,
                trace_id=trace.trace_id,
            )

        # ==============================================================
        # Step 2.5: Query Filter & Scope Understanding (Active vs Archived)
        # ==============================================================
        target_status, target_policy_id, detected_type = self.scope_checker.detect_query_filters(contextualized_query)
        effective_policy_type = policy_type or detected_type

        # ==============================================================
        # Step 3: Semantic Cache Lookup
        # ==============================================================
        settings = get_settings()
        query_vector: Optional[List[float]] = None
        cache_hit_entry = None
        active_cache_version = f"{getattr(settings, 'current_policy_version', '2026')}_{target_status}"

        with trace.measure_step("cache") as step_meta:
            if getattr(settings, "cache_enabled", True):
                try:
                    query_vector = self.embedder.embed_text(contextualized_query)
                    cache_hit_entry = self.cache.lookup(
                        query_vector=query_vector,
                        current_version=active_cache_version,
                    )
                    if cache_hit_entry is None and target_status == "active":
                        cache_hit_entry = self.cache.lookup(
                            query_vector=query_vector,
                            current_version=getattr(settings, "current_policy_version", "2026"),
                        )
                    step_meta["cache_hit"] = cache_hit_entry is not None
                except Exception as exc:
                    logger.warning(f"Semantic cache lookup encountered error: {exc}. Proceeding to search.")
                    step_meta["cache_error"] = str(exc)
            else:
                step_meta["cache_enabled"] = False

        if cache_hit_entry is not None:
            # Update conversation memory with cached exchange
            session_mem.add_turn(sanitized_query, cache_hit_entry.answer)
            trace.final_answer = cache_hit_entry.answer
            trace.model_used = "semantic-cache"
            self.tracer.finish_trace(trace)
            return AgentResponse(
                answer=cache_hit_entry.answer,
                sources=cache_hit_entry.sources,
                cache_hit=True,
                in_scope=True,
                model_used="semantic-cache",
                session_id=sid,
                trace_id=trace.trace_id,
            )

        # ==============================================================
        # Step 4 & 5: Controlled Search (Retrieval, Fusion, Reranking)
        # ==============================================================
        from app.rag.multilingual import translate_query_to_english_for_search

        search_query = translate_query_to_english_for_search(
            contextualized_query, active_language=language
        )

        search_result: PolicySearchResult = self.search_tool.execute(
            query=search_query,
            policy_type=effective_policy_type,
            status=target_status,
            policy_id=target_policy_id,
            top_k=top_k,
            candidate_k=candidate_k,
            trace=trace,
        )

        raw_passages = search_result.passages
        trace.retrieved_documents = list(dict.fromkeys(p.document for p in raw_passages))

        if not raw_passages:
            no_info_msg = "I couldn't find this information in the available XYZ Technologies HR policy documents."
            session_mem.add_turn(sanitized_query, no_info_msg)
            trace.final_answer = no_info_msg
            self.tracer.finish_trace(trace)
            return AgentResponse(
                answer=no_info_msg,
                in_scope=True,
                session_id=sid,
                reranker_used=search_result.reranker_used,
                trace_id=trace.trace_id,
            )

        # ==============================================================
        # Security Pipeline Stage 2: Screen Retrieved Untrusted Documents
        # ==============================================================
        safe_passages = []
        for p in raw_passages:
            doc_screen = self.model_armor.screen_document(p.text, p.document)
            if not doc_screen.is_safe:
                logger.warning(
                    f"Indirect prompt injection neutralized in document '{p.document}': {doc_screen.threats_detected}"
                )
                p_safe = p.model_copy(update={"text": doc_screen.sanitized_text})
                safe_passages.append(p_safe)
            else:
                safe_passages.append(p)

        # ==============================================================
        # Step 6: Generate Grounded Answer via LiteLLM Router
        # ==============================================================
        citations: List[SourceCitation] = extract_citations_from_results(safe_passages)
        sources_payload = [c.to_ui_dict() for c in citations]

        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(contextualized_query, safe_passages, language=language)

        # Append recent history context if multi-turn dialogue exists
        history_context = session_mem.get_recent_context_str(max_turns=3)
        if history_context:
            user_prompt = f"Recent Dialogue History:\n{history_context}\n\n{user_prompt}"

        model_used = "unknown"
        fallback_used = False
        raw_answer = ""

        with trace.measure_step("llm") as step_meta:
            try:
                active_llm = self.llm or get_llm_service()
                raw_answer = active_llm.generate(prompt=user_prompt, system_prompt=system_prompt)

                last_resp = getattr(active_llm, "last_response", None)
                if last_resp is not None:
                    model_used = last_resp.model_used
                    fallback_used = last_resp.fallback_used
                    step_meta["model"] = model_used
                    step_meta["fallback"] = fallback_used
                    if getattr(last_resp, "usage", None):
                        trace.token_usage = {
                            "prompt_tokens": last_resp.usage.get("prompt_tokens", 0),
                            "completion_tokens": last_resp.usage.get("completion_tokens", 0),
                            "total_tokens": last_resp.usage.get("total_tokens", 0),
                        }
                    if fallback_used and getattr(settings, "google_cloud_project", None):
                        raw_answer += f"\n\n*Note: Response generated using backup model ({model_used}) due to primary service unavailability.*"
                else:
                    m_name = getattr(active_llm, "model_name", "unknown")
                    model_used = m_name if isinstance(m_name, str) else "mock-model"
                    step_meta["model"] = model_used

            except LLMConfigurationError as llm_cfg_err:
                logger.warning(f"LLM not configured: {llm_cfg_err}")
                top_doc = safe_passages[0]
                model_used = "fallback-excerpt"
                step_meta["model"] = model_used
                step_meta["unconfigured"] = True
                raw_answer = (
                    f"**Policy Excerpt Found** (Top Match: {top_doc.document}, Section: {top_doc.section}, Page: {top_doc.page}):\n\n"
                    f"> {top_doc.text}\n\n"
                    f"*Note: LLM synthesis is inactive because credentials are not configured: {llm_cfg_err}*"
                )
            except Exception as exc:
                logger.error(f"LLM generation failed: {exc}")
                step_meta["error"] = str(exc)
                trace.errors.append(f"[llm] {exc}")
                raw_answer = f"An error occurred during response generation: {exc}"

        trace.model_used = model_used
        trace.fallback_used = fallback_used

        # ==============================================================
        # Security Pipeline Stage 3: Output Validation & Model Armor Screening
        # ==============================================================
        with trace.measure_step("output_security") as step_meta:
            output_screening = self.model_armor.screen_output(raw_answer, retrieved_passages=safe_passages)
            step_meta["is_safe"] = output_screening.is_safe
            step_meta["threats_detected"] = output_screening.threats_detected

        final_answer = output_screening.sanitized_text

        # ==============================================================
        # Step 7: Store in Cache & Memory, Return Response
        # ==============================================================
        if getattr(settings, "cache_enabled", True) and query_vector is not None:
            try:
                self.cache.store(
                    question=contextualized_query,
                    embedding=query_vector,
                    answer=final_answer,
                    sources=sources_payload,
                    policy_version=active_cache_version,
                )
            except Exception as exc:
                logger.warning(f"Failed to persist entry to semantic cache: {exc}")

        session_mem.add_turn(sanitized_query, final_answer)

        # Finalize and record trace
        trace.final_answer = final_answer
        self.tracer.finish_trace(trace)

        return AgentResponse(
            answer=final_answer,
            sources=sources_payload,
            retrieval_results=[p.model_dump() for p in safe_passages],
            cache_hit=False,
            in_scope=True,
            model_used=model_used,
            fallback_used=fallback_used,
            reranker_used=search_result.reranker_used,
            session_id=sid,
            trace_id=trace.trace_id,
        )


# Global agent instance
_GLOBAL_AGENT: Optional[HRPolicyAgent] = None

# Aliases
GuardedHRAgent = HRPolicyAgent


def get_agent() -> HRPolicyAgent:
    """Return singleton HRPolicyAgent instance."""
    global _GLOBAL_AGENT
    if _GLOBAL_AGENT is None:
        _GLOBAL_AGENT = HRPolicyAgent()
    return _GLOBAL_AGENT


def reset_agent() -> None:
    """Reset singleton HRPolicyAgent instance to refresh search indices."""
    global _GLOBAL_AGENT
    _GLOBAL_AGENT = None


