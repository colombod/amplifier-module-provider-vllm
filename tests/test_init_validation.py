"""Tests for VLLMProvider initialization validation.

Ensures that VLLMProvider fails fast at construction time when neither
base_url nor client is provided, so the CLI's _try_instantiate_provider()
Approach 1 (no base_url) correctly falls through to Approach 3 (with base_url).
"""

import pytest
from unittest.mock import MagicMock

from amplifier_module_provider_vllm import VLLMProvider


class TestVLLMProviderInit:
    def test_raises_without_base_url_or_client(self):
        """VLLMProvider should raise ValueError when neither base_url nor client is provided."""
        with pytest.raises(ValueError, match="base_url or client must be provided"):
            VLLMProvider(api_key="test", config={})

    def test_raises_with_none_base_url_and_no_client(self):
        """VLLMProvider should raise ValueError when base_url is explicitly None."""
        with pytest.raises(ValueError, match="base_url or client must be provided"):
            VLLMProvider(base_url=None, api_key="test", config={})

    def test_succeeds_with_base_url(self):
        """VLLMProvider should succeed when base_url is provided."""
        provider = VLLMProvider(base_url="http://localhost:8000/v1", config={})
        assert provider.base_url == "http://localhost:8000/v1"

    def test_succeeds_with_client(self):
        """VLLMProvider should succeed when client is provided."""
        mock_client = MagicMock()
        provider = VLLMProvider(client=mock_client, config={})
        assert provider._client is mock_client
