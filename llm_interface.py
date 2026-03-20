"""
Service Layer — Ollama LLM Streaming Client

Responsibility: Communicate with the local Ollama server to generate
streaming LLM responses. Uses httpx for HTTP requests.

Permitted imports: Python stdlib, httpx,
    infrastructure layer (config, logger, exceptions).
Must NOT import: streamlit, rag_pipeline, app, or any UI/pipeline module.
"""

from typing import Generator, Optional

import httpx

from config import get_config
from exceptions import OllamaConnectionError
from logger import get_logger

logger = get_logger(__name__)


class OllamaClient:
    """Lazy singleton HTTP client for the Ollama LLM server.

    Connects to the local Ollama instance for text generation.
    The HTTP client is created on first use.
    """

    _instance: Optional["OllamaClient"] = None
    _client: Optional[httpx.Client] = None

    def __init__(self) -> None:
        """Private init — use OllamaClient.get() instead."""

    @classmethod
    def get(cls) -> "OllamaClient":
        """Return the singleton OllamaClient.

        Returns:
            The singleton OllamaClient instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        if cls._client is not None:
            try:
                cls._client.close()
            except Exception:
                pass
        cls._instance = None
        cls._client = None

    def _ensure_client(self) -> httpx.Client:
        """Create the HTTP client if not yet initialized.

        Returns:
            The httpx.Client instance.
        """
        if self._client is None:
            config = get_config()
            # Ollama HTTP client — no significant RAM, just a connection
            logger.info(
                "[LAZY] Loading Ollama HTTP client for the first time "
                "(base_url=%s)",
                config.ollama_base_url,
            )
            self._client = httpx.Client(
                base_url=config.ollama_base_url,
                timeout=httpx.Timeout(
                    connect=5.0,
                    read=120.0,
                    write=10.0,
                    pool=10.0,
                ),
            )
        return self._client

    def health_check(self) -> bool:
        """Check if the Ollama server is reachable and responding.

        Returns:
            True if Ollama is reachable, False otherwise.
        """
        try:
            client = self._ensure_client()
            response = client.get("/api/tags")
            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            logger.warning("Ollama health check failed: %s", e)
            return False
        except Exception as e:
            logger.warning("Ollama health check unexpected error: %s", e)
            return False

    def stream_response(self, prompt: str) -> Generator[str, None, None]:
        """Stream a response from Ollama token by token.

        Args:
            prompt: The full prompt to send to the LLM.

        Yields:
            String tokens as they are generated.

        Raises:
            OllamaConnectionError: If Ollama is unreachable or returns an error.
        """
        config = get_config()
        client = self._ensure_client()

        payload = {
            "model": config.ollama_model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": config.ollama_temperature,
                "num_predict": config.ollama_max_tokens,
            },
        }

        try:
            with client.stream(
                "POST", "/api/generate", json=payload
            ) as response:
                if response.status_code != 200:
                    raise OllamaConnectionError(
                        config.ollama_base_url,
                        f"HTTP {response.status_code}",
                    )

                import json

                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

        except OllamaConnectionError:
            raise
        except httpx.ConnectError as e:
            raise OllamaConnectionError(
                config.ollama_base_url, f"Connection refused: {e}"
            ) from e
        except httpx.TimeoutException as e:
            raise OllamaConnectionError(
                config.ollama_base_url, f"Request timed out: {e}"
            ) from e
        except Exception as e:
            raise OllamaConnectionError(
                config.ollama_base_url, f"Unexpected error: {e}"
            ) from e

    def generate(self, prompt: str) -> str:
        """Generate a complete (non-streaming) response from Ollama.

        Args:
            prompt: The full prompt to send to the LLM.

        Returns:
            The complete generated text.

        Raises:
            OllamaConnectionError: If Ollama is unreachable.
        """
        tokens = list(self.stream_response(prompt))
        return "".join(tokens)
