"""Tests for the llm_provider module."""

import os
from unittest import mock

import pytest
import requests

from llm_markdown_generator.error_handler import (
    AuthError,
    NetworkError,
    ParsingError,
    RateLimitError,
    RetryConfig,
    ServiceUnavailableError,
    TimeoutError
)
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
            with pytest.raises(AuthError):
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
                retry_config=RetryConfig(max_retries=2, base_delay=0.1),
            )

            assert provider.model_name == "gpt-4"
            assert provider.api_key == "test-api-key"
            assert provider.temperature == 0.7
            assert provider.max_tokens == 500
            assert provider.additional_params == {"top_p": 0.9}
            assert provider.retry_config.max_retries == 2
            assert provider.retry_config.base_delay == 0.1

    def test_openai_provider_generate_text_success(self):
        """Test successful text generation with OpenAI provider."""
        # Mock response data
        mock_response = mock.Mock()
        mock_response.status_code = 200
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
                    # Disable retries for this test
                    retry_config=RetryConfig(max_retries=0),
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
                "requests.post", side_effect=requests.ConnectionError("Connection error")
            ):
                provider = OpenAIProvider(
                    model_name="gpt-4",
                    api_key_env_var="OPENAI_API_KEY",
                    temperature=0.7,
                    # Disable retries for this test
                    retry_config=RetryConfig(max_retries=0),
                )

                with pytest.raises(NetworkError) as exc_info:
                    provider.generate_text("Test prompt")

                assert "Network error connecting to OpenAI API" in str(exc_info.value)

    def test_openai_provider_generate_text_timeout_error(self):
        """Test handling of timeout errors during text generation."""
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
            with mock.patch(
                "requests.post", side_effect=requests.Timeout("Request timed out")
            ):
                provider = OpenAIProvider(
                    model_name="gpt-4",
                    api_key_env_var="OPENAI_API_KEY",
                    temperature=0.7,
                    # Disable retries for this test
                    retry_config=RetryConfig(max_retries=0),
                )

                with pytest.raises(TimeoutError) as exc_info:
                    provider.generate_text("Test prompt")

                assert "Request to OpenAI API timed out" in str(exc_info.value)

    def test_openai_provider_generate_text_response_error(self):
        """Test handling of response parsing errors during text generation."""
        # Mock response with missing expected data
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"invalid": "response format"}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
            with mock.patch("requests.post", return_value=mock_response):
                provider = OpenAIProvider(
                    model_name="gpt-4",
                    api_key_env_var="OPENAI_API_KEY",
                    temperature=0.7,
                    # Disable retries for this test
                    retry_config=RetryConfig(max_retries=0),
                )

                with pytest.raises(ParsingError) as exc_info:
                    provider.generate_text("Test prompt")

                assert "Error parsing OpenAI API response" in str(exc_info.value)

    def test_openai_provider_generate_text_api_error(self):
        """Test handling of API errors during text generation."""
        # Mock response with error status and error details
        mock_response = mock.Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {
                "message": "Invalid request parameters",
                "type": "invalid_request_error",
                "code": "invalid_param"
            }
        }
        mock_response.text = '{"error": {"message": "Invalid request parameters"}}'
        mock_response.raise_for_status.side_effect = requests.HTTPError("400 Client Error")

        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
            with mock.patch("requests.post", return_value=mock_response):
                provider = OpenAIProvider(
                    model_name="gpt-4",
                    api_key_env_var="OPENAI_API_KEY",
                    temperature=0.7,
                    # Disable retries for this test
                    retry_config=RetryConfig(max_retries=0),
                )

                with pytest.raises(Exception) as exc_info:
                    provider.generate_text("Test prompt")

                assert "Invalid request" in str(exc_info.value)
                
    def test_openai_provider_retry_mechanism(self):
        """Test retry mechanism for transient errors."""
        # Create a side effect sequence: first request fails with a retryable error,
        # second request succeeds
        mock_success_response = mock.Mock()
        mock_success_response.status_code = 200
        mock_success_response.json.return_value = {
            "choices": [
                {"message": {"content": "Retry successful response"}}
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        mock_success_response.raise_for_status = mock.Mock()
        
        # Mock post with side effects: first a rate limit error, then success
        mock_post = mock.Mock()
        mock_post.side_effect = [
            requests.exceptions.RequestException("Rate limit exceeded"),
            mock_success_response
        ]
        
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
            with mock.patch("requests.post", mock_post):
                provider = OpenAIProvider(
                    model_name="gpt-4",
                    api_key_env_var="OPENAI_API_KEY",
                    temperature=0.7,
                    # Configure retries with very small delay for testing
                    retry_config=RetryConfig(max_retries=2, base_delay=0.01),
                )
                
                with mock.patch("time.sleep") as mock_sleep:  # Skip actual sleep to speed up test
                    result = provider.generate_text("Test prompt")
                
                assert result == "Retry successful response"
                assert mock_post.call_count == 2  # Initial call fails, retry succeeds
                assert mock_sleep.call_count == 1  # Should sleep once before retry
                
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
        with pytest.raises(AuthError):
            GeminiProvider(
                model_name="gemini-1.5-flash",
                temperature=0.7,
            )

        # Empty environment variable
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            with pytest.raises(AuthError):
                GeminiProvider(
                    model_name="gemini-1.5-flash",
                    api_key_env_var="GEMINI_API_KEY",
                    temperature=0.7,
                )

    def test_gemini_provider_generate_text_success(self):
        """Test successful text generation with Gemini provider."""
        # Mock response data
        mock_response = mock.Mock()
        mock_response.status_code = 200
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
                # Disable retries for this test
                retry_config=RetryConfig(max_retries=0),
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
            "requests.post", side_effect=requests.ConnectionError("Connection error")
        ):
            provider = GeminiProvider(
                model_name="gemini-1.5-flash",
                api_key="test-api-key",
                temperature=0.7,
                # Disable retries for this test
                retry_config=RetryConfig(max_retries=0),
            )

            with pytest.raises(NetworkError) as exc_info:
                provider.generate_text("Test prompt")

            assert "Network error connecting to Gemini API" in str(exc_info.value)

    def test_gemini_provider_generate_text_response_error(self):
        """Test handling of response parsing errors during Gemini text generation."""
        # Mock response with missing expected data
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"invalid": "response format"}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("requests.post", return_value=mock_response):
            provider = GeminiProvider(
                model_name="gemini-1.5-flash",
                api_key="test-api-key",
                temperature=0.7,
                # Disable retries for this test
                retry_config=RetryConfig(max_retries=0),
            )

            with pytest.raises(ParsingError) as exc_info:
                provider.generate_text("Test prompt")

            assert "No response candidates" in str(exc_info.value)

    def test_gemini_provider_generate_text_api_error(self):
        """Test handling of API errors during Gemini text generation."""
        # Mock response with error status and error message
        mock_response = mock.Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {
            "error": {"message": "Invalid API key"}
        }
        mock_response.text = '{"error": {"message": "Invalid API key"}}'
        
        with mock.patch("requests.post", return_value=mock_response):
            provider = GeminiProvider(
                model_name="gemini-1.5-flash",
                api_key="test-api-key",
                temperature=0.7,
                # Disable retries for this test
                retry_config=RetryConfig(max_retries=0),
            )

            with pytest.raises(AuthError) as exc_info:
                provider.generate_text("Test prompt")

            assert "Authentication error" in str(exc_info.value)