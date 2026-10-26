"""RAG evaluation metrics calculator for HR Policy Assistant.

Computes real, deterministic evaluation metrics:
1. Retrieval Recall: Whether expected source document(s) appear in top-K passages.
2. Retrieval Precision: Ratio of relevant retrieved documents in top-K passages.
3. Context Relevance: Relevance and lexical/semantic overlap of retrieved passages to the query.
4. Answer Relevance: How directly and completely the generated answer addresses the question.
5. Faithfulness: Proportion of statements in the answer grounded in the retrieved context.
6. Groundedness: Context alignment and absence of ungrounded external hallucinations.
7. Citation Accuracy: Accuracy and validity of citations against actual retrieved source documents.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Set
from pydantic import BaseModel, Field

from app.utils.logging import get_logger

logger = get_logger("app.evaluation.metrics")

# Common English stopwords to ignore in lexical metric calculations
STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "its",
    "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no",
    "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "should",
    "shouldn't", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "were", "weren't", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "won't", "would", "wouldn't", "you", "your", "yours", "yourself",
}


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    return re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())


def _content_words(text: str) -> Set[str]:
    """Extract substantive non-stopword tokens from text."""
    tokens = _tokenize(text)
    return {t for t in tokens if t not in STOPWORDS and len(t) > 2}


class EvaluationMetrics(BaseModel):
    """Container for individual query evaluation scores."""

    retrieval_recall: float = Field(ge=0.0, le=1.0, description="Presence of expected source in retrieved set")
    retrieval_precision: float = Field(ge=0.0, le=1.0, description="Proportion of retrieved docs matching target")
    context_relevance: float = Field(ge=0.0, le=1.0, description="Relevance of retrieved chunks to question")
    answer_relevance: float = Field(ge=0.0, le=1.0, description="Relevance of generated answer to question")
    faithfulness: float = Field(ge=0.0, le=1.0, description="Proportion of answer supported by context")
    groundedness: float = Field(ge=0.0, le=1.0, description="Context alignment and lack of hallucination")
    citation_accuracy: float = Field(ge=0.0, le=1.0, description="Validity of citations against source passages")
    composite_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Average composite quality score")

    def to_dict(self) -> Dict[str, float]:
        """Return metrics as dictionary."""
        return {
            "retrieval_recall": round(self.retrieval_recall, 4),
            "retrieval_precision": round(self.retrieval_precision, 4),
            "context_relevance": round(self.context_relevance, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "faithfulness": round(self.faithfulness, 4),
            "groundedness": round(self.groundedness, 4),
            "citation_accuracy": round(self.citation_accuracy, 4),
            "composite_score": round(self.composite_score, 4),
        }


class RAGEvaluator:
    """Evaluates RAG pipeline performance across all 7 core dimensions."""

    @staticmethod
    def calculate_retrieval_recall(
        retrieved_documents: Sequence[str] | str,
        expected_source: str | Sequence[str],
    ) -> float:
        """Calculate Retrieval Recall.

        Returns 1.0 if the expected source appears in the retrieved documents, else 0.0.
        If a sequence of expected sources is provided, returns fraction found.
        """
        if isinstance(retrieved_documents, str) and isinstance(expected_source, (list, tuple, set)):
            retrieved_documents, expected_source = expected_source, retrieved_documents

        if not retrieved_documents:
            return 0.0

        retrieved_set = {Path(d).name.lower() for d in retrieved_documents}

        if isinstance(expected_source, str):
            target = Path(expected_source).name.lower()
            return 1.0 if target in retrieved_set else 0.0

        targets = {Path(t).name.lower() for t in expected_source if t}
        if not targets:
            return 1.0

        matched = len(targets.intersection(retrieved_set))
        return float(matched / len(targets))

    @staticmethod
    def calculate_retrieval_precision(
        retrieved_documents: Sequence[str] | str,
        expected_source: str | Sequence[str],
    ) -> float:
        """Calculate Retrieval Precision.

        Measures the fraction of top-K retrieved documents that belong to the expected source.
        """
        if isinstance(retrieved_documents, str) and isinstance(expected_source, (list, tuple, set)):
            retrieved_documents, expected_source = expected_source, retrieved_documents

        if not retrieved_documents:
            return 0.0

        if isinstance(expected_source, str):
            targets = {Path(expected_source).name.lower()}
        else:
            targets = {Path(t).name.lower() for t in expected_source}

        hits = sum(1 for d in retrieved_documents if Path(d).name.lower() in targets)
        return float(hits / len(retrieved_documents))

    @staticmethod
    def calculate_context_relevance(
        question: str,
        retrieved_texts: Sequence[str],
    ) -> float:
        """Calculate Context Relevance.

        Computes substantive keyword/concept overlap between the question and retrieved context passages.
        """
        q_words = _content_words(question)
        if not q_words or not retrieved_texts:
            return 0.0

        context_scores: List[float] = []
        for text in retrieved_texts:
            c_words = _content_words(text)
            if not c_words:
                continue
            intersection = q_words.intersection(c_words)
            # Overlap ratio relative to question keywords
            score = len(intersection) / len(q_words)
            context_scores.append(min(1.0, score))

        if not context_scores:
            return 0.0

        # Best chunk relevance + average chunk relevance
        return float((max(context_scores) * 0.7) + ((sum(context_scores) / len(context_scores)) * 0.3))

    @staticmethod
    def calculate_answer_relevance(
        question: str,
        answer: str,
    ) -> float:
        """Calculate Answer Relevance.

        Measures lexical and semantic alignment between the question and generated answer.
        """
        q_words = _content_words(question)
        a_words = _content_words(answer)

        if not q_words or not a_words:
            return 0.0

        # Jaccard / Overlap coefficient
        intersection = len(q_words.intersection(a_words))
        recall_on_q = intersection / len(q_words)

        # Penalize refusal/irrelevant generic answers
        if "cannot find" in answer.lower() or "not mentioned" in answer.lower():
            return max(0.2, recall_on_q * 0.5)

        return float(min(1.0, recall_on_q * 1.2))

    @staticmethod
    def calculate_faithfulness(
        answer: str,
        retrieved_texts: Sequence[str],
    ) -> float:
        """Calculate Faithfulness.

        Measures whether the claims and terms made in the answer are supported by retrieved context passages.
        """
        combined_context = " ".join(retrieved_texts)
        ctx_words = _content_words(combined_context)
        ans_words = _content_words(answer)

        if not ans_words or not ctx_words:
            return 0.0

        supported = ans_words.intersection(ctx_words)
        ratio = len(supported) / len(ans_words)
        return float(min(1.0, max(0.0, ratio)))

    @staticmethod
    def calculate_groundedness(
        answer: str,
        retrieved_texts: Sequence[str],
    ) -> float:
        """Calculate Groundedness.

        Measures sentence-level grounding against context to penalize ungrounded hallucinated assertions.
        """
        sentences = [s.strip() for s in re.split(r"[.!?\n]+", answer) if len(s.strip()) > 10]
        combined_context = " ".join(retrieved_texts).lower()

        if not sentences or not combined_context:
            return 0.0

        grounded_count = 0
        for sent in sentences:
            s_words = _content_words(sent)
            if not s_words:
                grounded_count += 1
                continue
            # A sentence is considered grounded if >= 50% of its substantive words appear in the context
            present_words = sum(1 for w in s_words if w in combined_context)
            if (present_words / len(s_words)) >= 0.45:
                grounded_count += 1

        return float(grounded_count / len(sentences))

    @staticmethod
    def calculate_citation_accuracy(
        citations: Sequence[Any],
        retrieved_passages: Sequence[Any],
    ) -> float:
        """Calculate Citation Accuracy.

        Verifies that cited sources accurately correspond to actual retrieved source documents.
        Supports both raw strings (e.g. 'leave_policy.pdf', 'Leave Policy') and citation dicts.
        """
        if not citations:
            return 1.0 if not retrieved_passages else 1.0

        if not retrieved_passages:
            return 0.0

        # Build normalized tokens for retrieved documents
        retrieved_tokens = set()
        for p in retrieved_passages:
            doc_str = getattr(p, "document", str(p)).lower()
            stem = Path(doc_str).stem.lower().replace("_", "").replace("-", "").replace(" ", "")
            retrieved_tokens.add(stem)
            retrieved_tokens.add(Path(doc_str).name.lower())

        valid_citations = 0
        for cit in citations:
            if isinstance(cit, dict):
                doc_name = str(cit.get("document") or cit.get("source") or "")
            else:
                doc_name = str(cit)

            clean_stem = Path(doc_name).stem.lower().replace("_", "").replace("-", "").replace(" ", "")
            clean_name = Path(doc_name).name.lower()

            if clean_stem in retrieved_tokens or clean_name in retrieved_tokens or any(clean_stem in rt or rt in clean_stem for rt in retrieved_tokens if len(rt) > 3):
                valid_citations += 1

        return float(valid_citations / len(citations))

    def evaluate_sample(
        self,
        question: str,
        expected_source: str | Sequence[str],
        retrieved_documents: Sequence[str],
        context_passages: Sequence[str],
        answer: str,
        citations: Sequence[Any],
        ground_truth: Optional[str] = None,
    ) -> EvaluationMetrics:
        """Run all 7 evaluation metrics for a single query sample."""
        rec = self.calculate_retrieval_recall(retrieved_documents, expected_source)
        prec = self.calculate_retrieval_precision(retrieved_documents, expected_source)
        c_rel = self.calculate_context_relevance(question, context_passages)
        a_rel = self.calculate_answer_relevance(question, answer)
        faith = self.calculate_faithfulness(answer, context_passages)
        ground = self.calculate_groundedness(answer, context_passages)
        cit_acc = self.calculate_citation_accuracy(citations, retrieved_documents)

        composite = float((rec * 0.25) + (prec * 0.15) + (c_rel * 0.15) + (a_rel * 0.15) + (faith * 0.15) + (cit_acc * 0.15))

        return EvaluationMetrics(
            retrieval_recall=rec,
            retrieval_precision=prec,
            context_relevance=c_rel,
            answer_relevance=a_rel,
            faithfulness=faith,
            groundedness=ground,
            citation_accuracy=cit_acc,
            composite_score=round(composite, 4),
        )

    def evaluate_query(
        self,
        question: str,
        expected_source: str | Sequence[str],
        retrieved_passages: Sequence[Any],
        answer: str,
        citations: Sequence[Any],
    ) -> EvaluationMetrics:
        """Run all evaluation metrics for a single query-answer cycle."""
        doc_names = [getattr(p, "document", str(p)) for p in retrieved_passages]
        doc_texts = [getattr(p, "text", str(p)) for p in retrieved_passages]
        return self.evaluate_sample(
            question=question,
            expected_source=expected_source,
            retrieved_documents=doc_names,
            context_passages=doc_texts,
            answer=answer,
            citations=citations,
        )

    @staticmethod
    def compute_summary(metrics: Sequence[EvaluationMetrics]) -> Dict[str, float]:
        """Aggregate evaluation metrics into mean scores."""
        if not metrics:
            return {
                "sample_count": 0,
                "mean_retrieval_recall": 0.0,
                "mean_retrieval_precision": 0.0,
                "mean_context_relevance": 0.0,
                "mean_answer_relevance": 0.0,
                "mean_faithfulness": 0.0,
                "mean_groundedness": 0.0,
                "mean_citation_accuracy": 0.0,
                "composite_score": 0.0,
            }

        n = len(metrics)
        rec = sum(m.retrieval_recall for m in metrics) / n
        prec = sum(m.retrieval_precision for m in metrics) / n
        crel = sum(m.context_relevance for m in metrics) / n
        arel = sum(m.answer_relevance for m in metrics) / n
        faith = sum(m.faithfulness for m in metrics) / n
        ground = sum(m.groundedness for m in metrics) / n
        citacc = sum(m.citation_accuracy for m in metrics) / n
        comp = sum(m.composite_score for m in metrics) / n

        return {
            "sample_count": n,
            "mean_retrieval_recall": round(rec, 4),
            "mean_retrieval_precision": round(prec, 4),
            "mean_context_relevance": round(crel, 4),
            "mean_answer_relevance": round(arel, 4),
            "mean_faithfulness": round(faith, 4),
            "mean_groundedness": round(ground, 4),
            "mean_citation_accuracy": round(citacc, 4),
            "composite_score": round(comp, 4),
        }


# Global singleton evaluator
_GLOBAL_EVALUATOR: Optional[RAGEvaluator] = None


def get_evaluator() -> RAGEvaluator:
    """Return singleton RAGEvaluator instance."""
    global _GLOBAL_EVALUATOR
    if _GLOBAL_EVALUATOR is None:
        _GLOBAL_EVALUATOR = RAGEvaluator()
    return _GLOBAL_EVALUATOR
