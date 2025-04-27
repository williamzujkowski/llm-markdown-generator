"""LLM Provider interfaces and implementations.

Defines the abstract base class for LLM providers and implements
specific provider clients (OpenAI and Google Gemini).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import json
import logging
import os
import requests

from llm_markdown_generator.error_handler import (
    AuthError, 
    classify_error, 
    ContentFilterError, 
    ContextLengthError, 
    InvalidRequestError, 
    LLMErrorBase, 
    NetworkError, 
    ParsingError,
    RateLimitError, 
    retry_with_backoff, 
    RetryConfig, 
    ServiceUnavailableError, 
    TimeoutError
)

# Set up logging
logger = logging.getLogger(__name__)

# For backward compatibility
LLMError = LLMErrorBase


# Token tracking functionality has been removed


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    This defines the interface that all LLM provider implementations must follow.
    """
    
    def __init__(self) -> None:
        """Initialize the LLM provider."""
        pass
    
    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        """Generate text from the LLM based on the provided prompt.

        Args:
            prompt: The prompt to send to the LLM.

        Returns:
            str: The generated text response.

        Raises:
            LLMError: If there is an error generating text.
        """
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation.

    Uses the OpenAI API to generate text responses.
    """

    def __init__(
        self,
        model_name: str,
        api_key_env_var: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        additional_params: Optional[Dict[str, Any]] = None,
        retry_config: Optional[RetryConfig] = None,
    ) -> None:
        """Initialize the OpenAI provider.

        Args:
            model_name: The name of the OpenAI model to use.
            api_key_env_var: The name of the environment variable containing the API key.
            temperature: Controls randomness in the output. Higher values (closer to 1)
                         mean more random, lower values (closer to 0) mean more deterministic.
            max_tokens: The maximum number of tokens to generate. If None, uses the model default.
            additional_params: Additional parameters to pass to the OpenAI API.
            retry_config: Configuration for retry behavior. If None, default settings are used.

        Raises:
            AuthError: If the API key is not set in the environment.
        """
        super().__init__()
        
        self.model_name = model_name
        self.api_key = os.environ.get(api_key_env_var)
        if not self.api_key:
            raise AuthError(f"API key not found in environment variable {api_key_env_var}")

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.additional_params = additional_params or {}
        self.api_base = "https://api.openai.com/v1/chat/completions"
        self.retry_config = retry_config or RetryConfig()
        
        logger.debug(f"Initialized OpenAI provider with model: {model_name}")

    def generate_text(self, prompt: str) -> str:
        """Generate text using the OpenAI API.

        Args:
            prompt: The prompt to send to the OpenAI API.

        Returns:
            str: The generated text response.

        Raises:
            Various LLMErrorBase subclasses depending on the type of error:
            - AuthError: Authentication or authorization error
            - RateLimitError: API rate limit exceeded
            - NetworkError: Network connectivity issue
            - TimeoutError: Request timed out
            - ContextLengthError: Prompt exceeded model's context length
            - ContentFilterError: Content was filtered by the API's safety systems
            - InvalidRequestError: Invalid request parameters
            - ParsingError: Error parsing the API response
            - LLMErrorBase: Other/unknown errors
        """
        # Use the retry mechanism for the actual API request
        try:
            return retry_with_backoff(
                self._generate_text_internal,
                retry_config=self.retry_config,
                on_retry=self._log_retry_attempt,
                prompt=prompt
            )
        except Exception as e:
            # If the exception is already an LLMErrorBase type, re-raise it
            if isinstance(e, LLMErrorBase):
                raise
            
            # Otherwise, classify and wrap the error
            error_type = classify_error(e)
            raise error_type(f"Error with OpenAI API: {str(e)}", raw_error=e)
    
    def _log_retry_attempt(self, attempt: int, exception: Exception, next_delay: float) -> None:
        """Log information about a retry attempt.
        
        Args:
            attempt: Current retry attempt number
            exception: The exception that triggered the retry
            next_delay: The delay before the next retry in seconds
        """
        logger.warning(
            f"OpenAI API request failed (attempt {attempt}). "
            f"Retrying in {next_delay:.2f}s. Error: {str(exception)}"
        )
    
    def _generate_text_internal(self, prompt: str) -> str:
        """Internal method to make the actual API request.
        
        This method is wrapped by the retry mechanism.
        
        Args:
            prompt: The prompt to send to the OpenAI API.
            
        Returns:
            str: The generated text response.
            
        Raises:
            Various exceptions based on the type of error encountered.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }

        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        # Add any additional parameters
        payload.update(self.additional_params)

        try:
            logger.debug(f"Sending request to OpenAI API with model: {self.model_name}")
            response = requests.post(
                self.api_base, 
                headers=headers, 
                json=payload,
                timeout=30  # Add a default timeout
            )
            
            # Handle HTTP errors
            if response.status_code != 200:
                error_content = None
                try:
                    error_content = response.json()
                except:
                    pass
                
                # Categorize the error based on status code
                if response.status_code == 401 or response.status_code == 403:
                    raise AuthError(
                        f"Authentication error: {response.text}", 
                        status_code=response.status_code,
                        response=error_content
                    )
                elif response.status_code == 429:
                    # Extract retry-after header if present
                    retry_after = None
                    if "retry-after" in response.headers:
                        try:
                            retry_after = int(response.headers["retry-after"])
                        except ValueError:
                            pass
                            
                    raise RateLimitError(
                        f"Rate limit exceeded: {response.text}",
                        status_code=response.status_code,
                        response=error_content,
                        retry_after=retry_after
                    )
                elif response.status_code >= 500:
                    raise ServiceUnavailableError(
                        f"OpenAI server error ({response.status_code}): {response.text}",
                        status_code=response.status_code,
                        response=error_content
                    )
                else:
                    # Check for specific error types in the response
                    if error_content and "error" in error_content:
                        error_message = error_content["error"].get("message", "")
                        error_type = error_content["error"].get("type", "")
                        error_code = error_content["error"].get("code", "")
                        
                        # Look for context length errors
                        if any(kw in error_message.lower() for kw in ["maximum context length", "token limit"]):
                            raise ContextLengthError(
                                f"Context length exceeded: {error_message}",
                                status_code=response.status_code,
                                response=error_content
                            )
                        # Look for content filter issues
                        elif any(kw in error_message.lower() or kw in error_type.lower() 
                              for kw in ["content filter", "moderation", "policy", "flagged"]):
                            raise ContentFilterError(
                                f"Content was filtered: {error_message}",
                                status_code=response.status_code,
                                response=error_content
                            )
                        else:
                            raise InvalidRequestError(
                                f"Invalid request: {error_message}",
                                status_code=response.status_code,
                                response=error_content
                            )
                    else:
                        raise InvalidRequestError(
                            f"Request error ({response.status_code}): {response.text}",
                            status_code=response.status_code
                        )
            
            # Parse the successful response
            try:
                response_json = response.json()
                # Extract the generated text from the response
                generated_text = response_json["choices"][0]["message"]["content"]
                
                logger.debug(f"Successfully received response from OpenAI API")
                return generated_text.strip()
            except (KeyError, IndexError, ValueError, TypeError) as e:
                raise ParsingError(f"Error parsing OpenAI API response: {str(e)}", raw_error=e, response=response.json())

        except requests.Timeout as e:
            raise TimeoutError(f"Request to OpenAI API timed out: {str(e)}", raw_error=e)
            
        except requests.ConnectionError as e:
            raise NetworkError(f"Network error connecting to OpenAI API: {str(e)}", raw_error=e)
            
        except requests.RequestException as e:
            # Handle other request errors
            return self._handle_request_exception(e)
    
    def _handle_request_exception(self, exception: requests.RequestException) -> None:
        """Handle various RequestException types and convert to appropriate LLM errors.
        
        Args:
            exception: The requests exception to handle
            
        Raises:
            An appropriate LLMErrorBase subclass based on the exception
        """
        # Try to get status code and response from the exception
        status_code = None
        response_data = None
        
        if hasattr(exception, 'response') and exception.response is not None:
            status_code = exception.response.status_code
            try:
                response_data = exception.response.json()
            except:
                response_data = exception.response.text
        
        # Use the error classifier helper to determine the most appropriate error type
        error_type = classify_error(exception, response_data, status_code)
        
        # Special case for common OpenAI error patterns
        if response_data and isinstance(response_data, dict) and "error" in response_data:
            error_msg = response_data["error"].get("message", str(exception))
            error_type_str = response_data["error"].get("type", "")
            
            if "rate limit" in error_msg.lower() or "rate_limit" in error_type_str:
                # Extract retry-after if present
                retry_after = None
                if hasattr(exception, 'response') and exception.response is not None:
                    if "retry-after" in exception.response.headers:
                        try:
                            retry_after = int(exception.response.headers["retry-after"])
                        except ValueError:
                            pass
                
                raise RateLimitError(
                    f"Rate limit exceeded: {error_msg}",
                    status_code=status_code,
                    response=response_data,
                    retry_after=retry_after,
                    raw_error=exception
                )
            
            # Use the default classifier for other cases
            raise error_type(
                f"Error with OpenAI API: {error_msg}",
                status_code=status_code,
                response=response_data,
                raw_error=exception
            )
        
        # Default handling if we couldn't get specific error details
        raise error_type(
            f"Error with OpenAI API: {str(exception)}",
            status_code=status_code,
            raw_error=exception
        )
            
# Token tracking functionality has been removed


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider implementation.
    
    Uses the Google Gemini API to generate text responses.
    """
    
    def __init__(
        self,
        model_name: str,
        api_key: str = None,
        api_key_env_var: str = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        additional_params: Optional[Dict[str, Any]] = None,
        retry_config: Optional[RetryConfig] = None,
    ) -> None:
        """Initialize the Gemini provider.
        
        Args:
            model_name: The name of the Gemini model to use (e.g., "gemini-1.5-flash").
            api_key: Direct API key for Gemini. If provided, takes precedence over api_key_env_var.
            api_key_env_var: The name of the environment variable containing the API key.
            temperature: Controls randomness in the output (0.0-1.0).
            max_tokens: The maximum number of tokens to generate.
            additional_params: Additional parameters to pass to the Gemini API.
            retry_config: Configuration for retry behavior. If None, default settings are used.
            
        Raises:
            AuthError: If no API key is provided or found in the environment.
        """
        super().__init__()
        
        self.model_name = model_name
        
        # Get API key either directly or from environment variable
        if api_key:
            self.api_key = api_key
        elif api_key_env_var:
            self.api_key = os.environ.get(api_key_env_var)
        else:
            raise AuthError("Either api_key or api_key_env_var must be provided")
            
        if not self.api_key:
            raise AuthError(f"API key not found in environment variable {api_key_env_var}")
        
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.additional_params = additional_params or {}
        self.api_base = f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent"
        self.retry_config = retry_config or RetryConfig()
        
        logger.debug(f"Initialized Gemini provider with model: {model_name}")
        
    def generate_text(self, prompt: str) -> str:
        """Generate text using the Google Gemini API.
        
        Args:
            prompt: The prompt to send to the Gemini API.
            
        Returns:
            str: The generated text response.
            
        Raises:
            Various LLMErrorBase subclasses depending on the type of error:
            - AuthError: Authentication or authorization error
            - RateLimitError: API rate limit exceeded
            - NetworkError: Network connectivity issue
            - TimeoutError: Request timed out
            - ContextLengthError: Prompt exceeded model's context length
            - ContentFilterError: Content was filtered by the API's safety systems
            - InvalidRequestError: Invalid request parameters
            - ParsingError: Error parsing the API response
            - LLMErrorBase: Other/unknown errors
        """
        # Use the retry mechanism for the actual API request
        try:
            return retry_with_backoff(
                self._generate_text_internal,
                retry_config=self.retry_config,
                on_retry=self._log_retry_attempt,
                prompt=prompt
            )
        except Exception as e:
            # If the exception is already an LLMErrorBase type, re-raise it
            if isinstance(e, LLMErrorBase):
                raise
            
            # Otherwise, classify and wrap the error
            error_type = classify_error(e)
            raise error_type(f"Error with Gemini API: {str(e)}", raw_error=e)
    
    def _log_retry_attempt(self, attempt: int, exception: Exception, next_delay: float) -> None:
        """Log information about a retry attempt.
        
        Args:
            attempt: Current retry attempt number
            exception: The exception that triggered the retry
            next_delay: The delay before the next retry in seconds
        """
        logger.warning(
            f"Gemini API request failed (attempt {attempt}). "
            f"Retrying in {next_delay:.2f}s. Error: {str(exception)}"
        )
        
    def _generate_text_internal(self, prompt: str) -> str:
        """Internal method to make the actual API request.
        
        This method is wrapped by the retry mechanism.
        
        Args:
            prompt: The prompt to send to the Gemini API.
            
        Returns:
            str: The generated text response.
            
        Raises:
            Various exceptions based on the type of error encountered.
        """
        # Construct the request URL with API key
        request_url = f"{self.api_base}?key={self.api_key}"
        
        # Prepare the headers
        headers = {
            "Content-Type": "application/json"
        }
        
        # Prepare the payload
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
            }
        }
        
        # Add max output tokens if specified
        if self.max_tokens is not None:
            payload["generationConfig"]["maxOutputTokens"] = self.max_tokens
            
        # Add any additional parameters
        if self.additional_params:
            payload["generationConfig"].update(self.additional_params)
            
        try:
            logger.debug(f"Sending request to Gemini API with model: {self.model_name}")
            response = requests.post(
                request_url, 
                headers=headers, 
                json=payload,
                timeout=30  # Add a default timeout
            )
            
            # Handle HTTP errors
            if response.status_code != 200:
                return self._handle_error_response(response)
            
            response_json = response.json()
            
            # Check for safety ratings blocking content
            if (
                "promptFeedback" in response_json and 
                response_json["promptFeedback"].get("blockReason", "") != ""
            ):
                block_reason = response_json["promptFeedback"]["blockReason"]
                safety_ratings = response_json["promptFeedback"].get("safetyRatings", [])
                raise ContentFilterError(
                    f"Content was filtered by Gemini safety system. Block reason: {block_reason}",
                    response=response_json
                )
                
            # Check for empty candidates (can happen with safety filtering)
            if "candidates" not in response_json or not response_json["candidates"]:
                if "promptFeedback" in response_json:
                    raise ContentFilterError(
                        "No response generated due to safety filtering",
                        response=response_json
                    )
                else:
                    raise ParsingError(
                        "No response candidates returned by Gemini API",
                        response=response_json
                    )
            
            # Extract the generated text from the response
            try:
                generated_text = response_json["candidates"][0]["content"]["parts"][0]["text"]
                
                logger.debug(f"Successfully received response from Gemini API")
                return generated_text.strip()
            except (KeyError, IndexError, ValueError, TypeError) as e:
                raise ParsingError(
                    f"Error parsing Gemini API response structure: {str(e)}",
                    raw_error=e,
                    response=response_json
                )
                
        except requests.Timeout as e:
            raise TimeoutError(f"Request to Gemini API timed out: {str(e)}", raw_error=e)
            
        except requests.ConnectionError as e:
            raise NetworkError(f"Network error connecting to Gemini API: {str(e)}", raw_error=e)
            
        except requests.RequestException as e:
            return self._handle_request_exception(e)
            
    def _handle_error_response(self, response: requests.Response) -> None:
        """Handle error responses from the Gemini API.
        
        Args:
            response: The error response from the API
            
        Raises:
            Appropriate LLMErrorBase subclass based on the error
        """
        # Try to parse error details from response
        error_content = None
        try:
            error_content = response.json()
        except:
            error_content = response.text
            
        error_message = "Unknown error"
        if isinstance(error_content, dict) and "error" in error_content:
            error_message = error_content["error"].get("message", "Unknown error")
            error_code = error_content["error"].get("code", 0)
            
            # Classify based on error message content
            if "API key" in error_message or "auth" in error_message.lower():
                raise AuthError(
                    f"Authentication error: {error_message}",
                    status_code=response.status_code,
                    response=error_content
                )
                
            elif "quota" in error_message.lower() or "rate" in error_message.lower():
                # Check for retry-after header
                retry_after = None
                if "retry-after" in response.headers:
                    try:
                        retry_after = int(response.headers["retry-after"])
                    except ValueError:
                        pass
                        
                raise RateLimitError(
                    f"Rate limit exceeded: {error_message}",
                    status_code=response.status_code,
                    response=error_content,
                    retry_after=retry_after
                )
                
            elif "content" in error_message.lower() and "safety" in error_message.lower():
                raise ContentFilterError(
                    f"Content was filtered: {error_message}",
                    status_code=response.status_code,
                    response=error_content
                )
                
            elif "token" in error_message.lower() and "limit" in error_message.lower():
                raise ContextLengthError(
                    f"Context length exceeded: {error_message}",
                    status_code=response.status_code,
                    response=error_content
                )
                
            elif response.status_code >= 500:
                raise ServiceUnavailableError(
                    f"Gemini server error ({response.status_code}): {error_message}",
                    status_code=response.status_code,
                    response=error_content
                )
                
            else:
                raise InvalidRequestError(
                    f"Invalid request: {error_message}",
                    status_code=response.status_code,
                    response=error_content
                )
        else:
            # Fallback if we couldn't parse structured error
            if response.status_code == 401 or response.status_code == 403:
                raise AuthError(
                    f"Authentication error ({response.status_code}): {response.text}",
                    status_code=response.status_code
                )
            elif response.status_code == 429:
                raise RateLimitError(
                    f"Rate limit exceeded ({response.status_code}): {response.text}",
                    status_code=response.status_code
                )
            elif response.status_code >= 500:
                raise ServiceUnavailableError(
                    f"Gemini server error ({response.status_code}): {response.text}",
                    status_code=response.status_code
                )
            else:
                raise InvalidRequestError(
                    f"Request error ({response.status_code}): {response.text}",
                    status_code=response.status_code
                )
                
    def _handle_request_exception(self, exception: requests.RequestException) -> None:
        """Handle various RequestException types and convert to appropriate LLM errors.
        
        Args:
            exception: The requests exception to handle
            
        Raises:
            An appropriate LLMErrorBase subclass based on the exception
        """
        # Try to get status code and response from the exception
        status_code = None
        response_data = None
        
        if hasattr(exception, 'response') and exception.response is not None:
            status_code = exception.response.status_code
            try:
                response_data = exception.response.json()
            except:
                response_data = exception.response.text
        
        # Special case for Gemini API error patterns
        if (
            response_data and 
            isinstance(response_data, dict) and 
            "error" in response_data
        ):
            error_msg = response_data["error"].get("message", str(exception))
            error_code = response_data["error"].get("code", 0)
            
            # Handle rate limiting specifically
            if "quota" in error_msg.lower() or "rate" in error_msg.lower():
                retry_after = None
                if hasattr(exception, 'response') and exception.response is not None:
                    if "retry-after" in exception.response.headers:
                        try:
                            retry_after = int(exception.response.headers["retry-after"])
                        except ValueError:
                            pass
                
                raise RateLimitError(
                    f"Rate limit exceeded: {error_msg}",
                    status_code=status_code,
                    response=response_data,
                    retry_after=retry_after,
                    raw_error=exception
                )
            
            # Handle authentication errors
            if "API key" in error_msg or "auth" in error_msg.lower():
                raise AuthError(
                    f"Authentication error: {error_msg}",
                    status_code=status_code,
                    response=response_data,
                    raw_error=exception
                )
                
            # Use the generic error classifier for other cases
            error_type = classify_error(exception, response_data, status_code)
            raise error_type(
                f"Error with Gemini API: {error_msg}",
                status_code=status_code,
                response=response_data,
                raw_error=exception
            )
        
        # Use the generic error classifier as fallback
        error_type = classify_error(exception, response_data, status_code)
        raise error_type(
            f"Error with Gemini API: {str(exception)}",
            status_code=status_code,
            raw_error=exception
        )
            
# Token tracking functionality has been removed
