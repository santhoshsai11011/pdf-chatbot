"""Tests for the Ollama LLM interface service."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from exceptions import OllamaConnectionError
from llm_interface import OllamaClient


class TestOllamaClient:
    """Tests for the OllamaClient lazy singleton."""

    @patch("llm_interface.httpx.Client")
    def test_connection_error_when_unreachable(self, mock_client_cls):
        """Test that OllamaConnectionError is raised when server is down."""
        import httpx

        mock_client = MagicMock()
        mock_client.stream.side_effect = httpx.ConnectError(
            "Connection refused"
        )
        mock_client_cls.return_value = mock_client

        client = OllamaClient.get()

        with pytest.raises(OllamaConnectionError) as exc_info:
            list(client.stream_response("test prompt"))

        assert "Connection refused" in str(exc_info.value)

    @patch("llm_interface.httpx.Client")
    def test_health_check_failure(self, mock_client_cls):
        """Test health_check returns False when Ollama is down."""
        import httpx

        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.ConnectError("refused")
        mock_client_cls.return_value = mock_client

        client = OllamaClient.get()
        assert client.health_check() is False

    @patch("llm_interface.httpx.Client")
    def test_health_check_success(self, mock_client_cls):
        """Test health_check returns True when Ollama is running."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = OllamaClient.get()
        assert client.health_check() is True

    @patch("llm_interface.httpx.Client")
    def test_stream_response_tokens(self, mock_client_cls):
        """Test that streaming returns individual tokens."""
        # Create mock streaming response
        lines = [
            json.dumps({"response": "Hello", "done": False}),
            json.dumps({"response": " world", "done": False}),
            json.dumps({"response": "!", "done": True}),
        ]

        mock_stream_response = MagicMock()
        mock_stream_response.status_code = 200
        mock_stream_response.iter_lines.return_value = iter(lines)
        mock_stream_response.__enter__ = MagicMock(
            return_value=mock_stream_response
        )
        mock_stream_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_stream_response
        mock_client_cls.return_value = mock_client

        client = OllamaClient.get()
        tokens = list(client.stream_response("test"))

        assert tokens == ["Hello", " world", "!"]

    @patch("llm_interface.httpx.Client")
    def test_timeout_raises_connection_error(self, mock_client_cls):
        """Test that timeout raises OllamaConnectionError."""
        import httpx

        mock_client = MagicMock()
        mock_client.stream.side_effect = httpx.TimeoutException(
            "Request timed out"
        )
        mock_client_cls.return_value = mock_client

        client = OllamaClient.get()

        with pytest.raises(OllamaConnectionError) as exc_info:
            list(client.stream_response("test prompt"))

        assert "timed out" in str(exc_info.value).lower()

    def test_singleton_pattern(self):
        """Test that get() returns the same instance."""
        c1 = OllamaClient.get()
        c2 = OllamaClient.get()
        assert c1 is c2
