"""
Infrastructure Layer — Custom Exception Types

Responsibility: Define all custom exceptions used across the project.
Permitted imports: Python stdlib only. No internal project imports.

This is the single source of truth for all exception types.
No other module may define its own exception classes.
"""


class PDFLoadError(Exception):
    """Raised when a PDF file cannot be read or parsed."""

    def __init__(self, filepath: str, reason: str = "Unknown error") -> None:
        self.filepath = filepath
        self.reason = reason
        super().__init__(f"Failed to load PDF '{filepath}': {reason}")


class OllamaConnectionError(Exception):
    """Raised when the Ollama LLM service is unreachable or returns an error."""

    def __init__(self, base_url: str, reason: str = "Connection refused") -> None:
        self.base_url = base_url
        self.reason = reason
        super().__init__(
            f"Cannot connect to Ollama at '{base_url}': {reason}. "
            f"Ensure Ollama is running with: ollama serve"
        )


class LowConfidenceError(Exception):
    """Raised when retrieval confidence is below the configured threshold.

    Used for internal pipeline signalling. The pipeline catches this
    and returns a structured uncertainty response.
    """

    def __init__(self, score: float, threshold: float) -> None:
        self.score = score
        self.threshold = threshold
        super().__init__(
            f"Low retrieval confidence: {score:.3f} < threshold {threshold}"
        )


class VectorStoreError(Exception):
    """Raised when ChromaDB operations fail."""

    def __init__(self, operation: str, reason: str = "Unknown error") -> None:
        self.operation = operation
        self.reason = reason
        super().__init__(f"Vector store error during '{operation}': {reason}")


class ChunkingError(Exception):
    """Raised when text chunking fails."""

    def __init__(self, reason: str = "Unknown error") -> None:
        self.reason = reason
        super().__init__(f"Chunking error: {reason}")
