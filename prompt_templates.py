"""
Pipeline Layer — Prompt Builder

Responsibility: Construct structured prompts for the LLM, including
system instructions, few-shot examples, context chunks, and conversation
history.

Permitted imports: Python stdlib, infrastructure layer (config, logger).
Must NOT import: streamlit, app, or any UI module.
"""

from typing import List, Optional, Tuple

from logger import get_logger

logger = get_logger(__name__)

# System prompt enforcing answer-from-context-only behaviour
_SYSTEM_PROMPT = """You are a helpful document assistant. You answer questions \
based ONLY on the provided context from uploaded PDF documents.

Rules:
1. Only use information explicitly stated in the provided context.
2. If the context does not contain enough information to answer the question, \
say "I don't have enough information in the uploaded documents to answer this \
question."
3. Always cite which document and page number your answer comes from.
4. Be concise and direct in your answers.
5. Do not make assumptions or use external knowledge."""

# Few-shot examples demonstrating expected behavior
_FEW_SHOT_EXAMPLES = """
Example 1 (answerable):
Context: [Page 3, report.pdf] The company reported Q3 revenue of $4.2 billion, \
a 15% increase year-over-year. Operating margins improved to 22%.
Question: What was the Q3 revenue?
Answer: According to report.pdf (page 3), Q3 revenue was $4.2 billion, \
representing a 15% year-over-year increase.

Example 2 (unanswerable):
Context: [Page 1, manual.pdf] The device supports Bluetooth 5.0 and Wi-Fi 6. \
Battery capacity is 4500mAh.
Question: What is the device's screen resolution?
Answer: I don't have enough information in the uploaded documents to answer \
this question. The available context from manual.pdf (page 1) covers \
connectivity features and battery specifications, but does not mention \
screen resolution."""


def build_prompt(
    context_chunks: List[dict],
    question: str,
    history: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """Build a complete prompt with system instructions, context, and question.

    Args:
        context_chunks: List of dicts with 'text', 'source', 'page' keys.
        question: The user's question.
        history: Optional list of (question, answer) tuples for conversation
            context. Only the last 3 turns are included.

    Returns:
        The fully formatted prompt string.
    """
    parts: List[str] = []

    # System prompt
    parts.append(_SYSTEM_PROMPT)
    parts.append("")

    # Few-shot examples
    parts.append(_FEW_SHOT_EXAMPLES)
    parts.append("")

    # Document context
    parts.append("--- Document Context ---")
    if context_chunks:
        for chunk in context_chunks:
            source = chunk.get("source", "unknown")
            page = chunk.get("page", "?")
            text = chunk.get("text", "")
            parts.append(f"[Page {page}, {source}] {text}")
            parts.append("")
    else:
        parts.append("No relevant context found.")
        parts.append("")

    # Conversation history (last 3 turns)
    if history:
        recent = history[-3:]
        parts.append("--- Recent Conversation ---")
        for q, a in recent:
            parts.append(f"User: {q}")
            parts.append(f"Assistant: {a}")
            parts.append("")

    # Current question
    parts.append("--- Current Question ---")
    parts.append(f"Question: {question}")
    parts.append("")
    parts.append("Answer:")

    prompt = "\n".join(parts)
    logger.debug(
        "Built prompt with %d context chunks, %d history turns, %d chars",
        len(context_chunks),
        len(history) if history else 0,
        len(prompt),
    )
    return prompt
