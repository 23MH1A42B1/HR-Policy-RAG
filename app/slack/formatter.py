"""Slack Message and Citation Formatter for XYZ Technologies HR Policy Bot.

Constructs clean Slack Block Kit payloads and markdown text.
Strictly strips all technical retrieval metadata (chunk IDs, Qdrant vectors, BM25 scores).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def format_slack_sources_text(sources: List[Dict[str, Any]]) -> str:
    """Format sanitized citations into clean Slack markdown."""
    if not sources:
        return ""

    count = len(sources)
    lines = [f"\n*Sources ({count})*\n"]

    for idx, s in enumerate(sources, start=1):
        doc = s.get("document_name") or s.get("source") or "HR Policy"
        pid = s.get("policy_id") or "HR-POLICY"
        ver = s.get("version") or "2.0"
        sec = s.get("section")
        page = s.get("page")

        citation_lines = [f"📄 *{doc}*", f"`{pid}` • Version {ver}"]
        if sec and sec != "General":
            citation_lines.append(f"Section: {sec}")
        if page is not None:
            citation_lines.append(f"Page: {page}")

        lines.append("\n".join(citation_lines))
        if idx < count:
            lines.append("")  # Blank line separator between citations

    return "\n".join(lines)


def format_slack_response(
    answer: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    is_admin: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a Slack response dict containing fallback text and Block Kit blocks."""
    clean_answer = (answer or "").strip()
    sources = sources or []
    metadata = metadata or {}

    sources_text = format_slack_sources_text(sources)
    full_text = f"{clean_answer}\n\n{sources_text}".strip()

    blocks: List[Dict[str, Any]] = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": clean_answer,
            },
        }
    ]

    # Append Sources section using Block Kit if citations exist
    if sources:
        blocks.append({"type": "divider"})
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": sources_text.strip(),
                },
            }
        )

    # Optional admin-only diagnostic context
    if is_admin and metadata:
        model = metadata.get("model", "none")
        cache_hit = "Yes" if metadata.get("cache_hit") else "No"
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"🛠️ *Admin Diagnostics*: Model: `{model}` | Cache Hit: `{cache_hit}`",
                    }
                ],
            }
        )

    return {
        "text": full_text,
        "blocks": blocks,
    }
