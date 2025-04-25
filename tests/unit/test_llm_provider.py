"""Tests for the llm_provider module."""

import os
from unittest import mock

import pytest
import requests

from llm_markdown_generator.llm_provider import (
    GeminiProvider, 
    LLMError, 
    LLMProvider, 
    OpenAIProvider,
    TokenUsage
)


class TestTokenUsage:
    """Tests for the TokenUsage class."""
    
    def test_token_usage_init(self):
        """Test initialization of TokenUsage."""
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.cost is None
        
    def test_token_usage_with_values(self):
        """Test TokenUsage with specific values."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost=0.0123
        )
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.cost == 0.0123
        
    def test_token_usage_str(self):
        """Test string representation of TokenUsage."""
        # Without cost
        usage1 = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert "prompt: 100" in str(usage1)
        assert "completion: 50" in str(usage1)
        assert "total: 150" in str(usage1)
        assert "cost" not in str(usage1)
        
        # With cost
        usage2 = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150, cost=0.0123)
        assert "prompt: 100" in str(usage2)
        assert "completion: 50" in str(usage2)
        assert "total: 150" in str(usage2)
        assert "cost: $0.012300" in str(usage2)


class TestLLMProvider:
    """Tests for the LLMProvider classes."""

    def test_openai_provider_init_missing_api_key(self):
        """Test OpenAI provider initialization with missing API key."""
        # Ensure environment variable is not set
        with mock.patch.dict(os.environ, clear=True):
            with pytest.raises(LLMError):
                OpenAIProvider(
                    model_name="gpt-4",
                    api_key_env_var="OPENAI_API_KEY",
                    temperature=0.7,
                )

    def test_openai_provider_init_success(self):
        """Test successful OpenAI provider initialization."""
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
            provider = OpenAIProvider(
                model_name="gpt-4",
                api_key_env_var="OPENAI_API_KEY",
                temperature=0.7,
                max_tokens=500,
                additional_params={"top_p": 0.9},
            )

            assert provider.model_name == "gpt-4"
            assert provider.api_key == "test-api-key"
            assert provider.temperature == 0.7
            assert provider.max_tokens == 500
            assert provider.additional_params == {"top_p": 0.9}

    def test_openai_provider_generate_text_success(self):
        """Test successful text generation with OpenAI provider."""
        # Mock response data
        mock_response = mock.Mock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": " This is a test response."}}
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        mock_response.raise_for_status = mock.Mock()

        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
            with mock.patch("requests.post", return_value=mock_response):
                provider = OpenAIProvider(
                    model_name="gpt-4",
                    api_key_env_var="OPENAI_API_KEY",
                    temperature=0.7,
                )

                result = provider.generate_text("Test prompt")

                assert result == "This is a test response."
                
                # Check token usage
                usage = provider.get_token_usage()
                assert usage.prompt_tokens == 10
                assert usage.completion_tokens == 5
                assert usage.total_tokens == 15
                assert usage.cost is not None

    def test_openai_provider_generate_text_request_error(self):
        """Test handling of request errors during text generation."""
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
            with mock.patch(
                "requests.post", side_effect=requests.RequestException("Connection error")
            ):
                provider = OpenAIProvider(
                    model_name="gpt-4",
                    api_key_env_var="OPENAI_API_KEY",
                    temperature=0.7,
                )

                with pytest.raises(LLMError) as exc_info:
                    provider.generate_text("Test prompt")

                assert "Error calling OpenAI API" in str(exc_info.value)

    def test_openai_provider_generate_text_response_error(self):
        """Test handling of response parsing errors during text generation."""
        # Mock response with missing expected data
        mock_response = mock.Mock()
        mock_response.json.return_value = {"invalid": "response format"}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
            with mock.patch("requests.post", return_value=mock_response):
                provider = OpenAIProvider(
                    model_name="gpt-4",
                    api_key_env_var="OPENAI_API_KEY",
                    temperature=0.7,
                )

                with pytest.raises(LLMError) as exc_info:
                    provider.generate_text("Test prompt")

                assert "Error parsing OpenAI API response" in str(exc_info.value)

    def test_openai_provider_generate_text_api_error(self):
        """Test handling of API errors during text generation."""
        # Mock response with error status
        mock_response = mock.Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            "400 Client Error"
        )

        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
            with mock.patch("requests.post", return_value=mock_response):
                provider = OpenAIProvider(
                    model_name="gpt-4",
                    api_key_env_var="OPENAI_API_KEY",
                    temperature=0.7,
                )

                with pytest.raises(LLMError) as exc_info:
                    provider.generate_text("Test prompt")

                assert "Error calling OpenAI API" in str(exc_info.value)
                
    def test_gemini_provider_init_with_direct_api_key(self):
        """Test Gemini provider initialization with direct API key."""
        provider = GeminiProvider(
            model_name="gemini-1.5-flash",
            api_key="test-api-key",
            temperature=0.7,
            max_tokens=500,
            additional_params={"topK": 40},
        )

        assert provider.model_name == "gemini-1.5-flash"
        assert provider.api_key == "test-api-key"
        assert provider.temperature == 0.7
        assert provider.max_tokens == 500
        assert provider.additional_params == {"topK": 40}

    def test_gemini_provider_init_with_env_var(self):
        """Test Gemini provider initialization with environment variable."""
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-api-key"}):
            provider = GeminiProvider(
                model_name="gemini-1.5-flash",
                api_key_env_var="GEMINI_API_KEY",
                temperature=0.7,
            )

            assert provider.model_name == "gemini-1.5-flash"
            assert provider.api_key == "test-api-key"
            assert provider.temperature == 0.7

    def test_gemini_provider_init_missing_api_key(self):
        """Test Gemini provider initialization with missing API key."""
        # No API key provided
        with pytest.raises(LLMError):
            GeminiProvider(
                model_name="gemini-1.5-flash",
                temperature=0.7,
            )

        # Empty environment variable
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            with pytest.raises(LLMError):
                GeminiProvider(
                    model_name="gemini-1.5-flash",
                    api_key_env_var="GEMINI_API_KEY",
                    temperature=0.7,
                )

    def test_gemini_provider_generate_text_success(self):
        """Test successful text generation with Gemini provider."""
        # Mock response data
        mock_response = mock.Mock()
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "This is a test response from Gemini."}
                        ]
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 12,
                "candidatesTokenCount": 8
            }
        }
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("requests.post", return_value=mock_response):
            provider = GeminiProvider(
                model_name="gemini-1.5-flash",
                api_key="test-api-key",
                temperature=0.7,
            )

            result = provider.generate_text("Test prompt")

            assert result == "This is a test response from Gemini."
            
            # Check token usage
            usage = provider.get_token_usage()
            assert usage.prompt_tokens == 12
            assert usage.completion_tokens == 8
            assert usage.total_tokens == 20
            assert usage.cost is not None

    def test_gemini_provider_generate_text_request_error(self):
        """Test handling of request errors during Gemini text generation."""
        with mock.patch(
            "requests.post", side_effect=requests.RequestException("Connection error")
        ):
            provider = GeminiProvider(
                model_name="gemini-1.5-flash",
                api_key="test-api-key",
                temperature=0.7,
            )

            with pytest.raises(LLMError) as exc_info:
                provider.generate_text("Test prompt")

            assert "Error calling Gemini API" in str(exc_info.value)

    def test_gemini_provider_generate_text_response_error(self):
        """Test handling of response parsing errors during Gemini text generation."""
        # Mock response with missing expected data
        mock_response = mock.Mock()
        mock_response.json.return_value = {"invalid": "response format"}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("requests.post", return_value=mock_response):
            provider = GeminiProvider(
                model_name="gemini-1.5-flash",
                api_key="test-api-key",
                temperature=0.7,
            )

            with pytest.raises(LLMError) as exc_info:
                provider.generate_text("Test prompt")

            assert "Error parsing Gemini API response" in str(exc_info.value)

    def test_gemini_provider_generate_text_api_error(self):
        """Test handling of API errors during Gemini text generation."""
        # Mock response with error status and error message
        mock_response = mock.Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("400 Client Error")
        mock_response.json.return_value = {
            "error": {"message": "Invalid API key"}
        }
        
        # Create a mock response object with status_code and json method
        error_response = mock.Mock()
        error_response.json.return_value = {"error": {"message": "Invalid API key"}}
        
        # Create a RequestException with the response attribute
        mock_exception = requests.HTTPError("400 Client Error")
        mock_exception.response = error_response

        with mock.patch("requests.post", side_effect=mock_exception):
            provider = GeminiProvider(
                model_name="gemini-1.5-flash",
                api_key="test-api-key",
                temperature=0.7,
            )

            with pytest.raises(LLMError) as exc_info:
                provider.generate_text("Test prompt")

            assert "Error calling Gemini API" in str(exc_info.value)