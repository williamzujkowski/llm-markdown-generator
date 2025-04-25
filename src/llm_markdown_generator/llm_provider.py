"""LLM Provider interfaces and implementations.

Defines the abstract base class for LLM providers and implements
specific provider clients (OpenAI and Google Gemini).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import json
import os
import requests


class LLMError(Exception):
    """Raised when there is an error interacting with an LLM provider."""

    pass


@dataclass
class TokenUsage:
    """Token usage information for an LLM request-response.
    
    Attributes:
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        total_tokens: Total tokens used (prompt + completion).
        cost: Estimated cost in USD, if available.
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: Optional[float] = None
    
    def __str__(self) -> str:
        """Return a string representation of token usage."""
        cost_str = f", cost: ${self.cost:.6f}" if self.cost is not None else ""
        return (f"TokenUsage(prompt: {self.prompt_tokens}, "
                f"completion: {self.completion_tokens}, "
                f"total: {self.total_tokens}{cost_str})")


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    This defines the interface that all LLM provider implementations must follow.
    
    Attributes:
        total_usage: Accumulated token usage across all requests.
    """
    
    def __init__(self) -> None:
        """Initialize the LLM provider."""
        self.total_usage = TokenUsage()
    
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
    
    @abstractmethod
    def get_token_usage(self) -> TokenUsage:
        """Get token usage information for the most recent request.

        Returns:
            TokenUsage: Token usage information.
        """
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation.

    Uses the OpenAI API to generate text responses.
    """

    # OpenAI pricing per 1,000 tokens (as of 2025-04)
    # Source: https://openai.com/pricing
    PRICING = {
        # GPT-4 models
        "gpt-4": {"input": 0.03, "output": 0.06},  # Per 1K tokens
        "gpt-4-32k": {"input": 0.06, "output": 0.12},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        
        # GPT-3.5 models
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-3.5-turbo-16k": {"input": 0.0005, "output": 0.0015},
    }

    def __init__(
        self,
        model_name: str,
        api_key_env_var: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the OpenAI provider.

        Args:
            model_name: The name of the OpenAI model to use.
            api_key_env_var: The name of the environment variable containing the API key.
            temperature: Controls randomness in the output. Higher values (closer to 1)
                         mean more random, lower values (closer to 0) mean more deterministic.
            max_tokens: The maximum number of tokens to generate. If None, uses the model default.
            additional_params: Additional parameters to pass to the OpenAI API.

        Raises:
            LLMError: If the API key is not set in the environment.
        """
        super().__init__()
        
        self.model_name = model_name
        self.api_key = os.environ.get(api_key_env_var)
        if not self.api_key:
            raise LLMError(f"API key not found in environment variable {api_key_env_var}")

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.additional_params = additional_params or {}
        self.api_base = "https://api.openai.com/v1/chat/completions"
        
        # Token usage for the most recent request
        self.last_usage = TokenUsage()

    def generate_text(self, prompt: str) -> str:
        """Generate text using the OpenAI API.

        Args:
            prompt: The prompt to send to the OpenAI API.

        Returns:
            str: The generated text response.

        Raises:
            LLMError: If there is an error generating text.
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
            response = requests.post(self.api_base, headers=headers, json=payload)
            response.raise_for_status()
            response_json = response.json()

            # Extract the generated text from the response
            generated_text = response_json["choices"][0]["message"]["content"]
            
            # Extract token usage
            self._update_token_usage(response_json)
            
            return generated_text.strip()

        except requests.RequestException as e:
            raise LLMError(f"Error calling OpenAI API: {str(e)}")
        except (KeyError, IndexError) as e:
            raise LLMError(f"Error parsing OpenAI API response: {str(e)}")
            
    def _update_token_usage(self, response_json: Dict[str, Any]) -> None:
        """Update token usage information from the OpenAI API response.
        
        Args:
            response_json: The JSON response from the OpenAI API.
        """
        # Reset last usage
        self.last_usage = TokenUsage()
        
        # Extract token usage information if available
        if "usage" in response_json:
            usage = response_json["usage"]
            
            # Update last usage
            self.last_usage.prompt_tokens = usage.get("prompt_tokens", 0)
            self.last_usage.completion_tokens = usage.get("completion_tokens", 0)
            self.last_usage.total_tokens = usage.get("total_tokens", 0)
            
            # Calculate cost if pricing is available for this model
            if self.model_name in self.PRICING:
                pricing = self.PRICING[self.model_name]
                prompt_cost = (self.last_usage.prompt_tokens / 1000) * pricing["input"]
                completion_cost = (self.last_usage.completion_tokens / 1000) * pricing["output"]
                self.last_usage.cost = prompt_cost + completion_cost
            
            # Update total usage
            self.total_usage.prompt_tokens += self.last_usage.prompt_tokens
            self.total_usage.completion_tokens += self.last_usage.completion_tokens
            self.total_usage.total_tokens += self.last_usage.total_tokens
            
            if self.last_usage.cost is not None:
                if self.total_usage.cost is None:
                    self.total_usage.cost = 0
                self.total_usage.cost += self.last_usage.cost

    def get_token_usage(self) -> TokenUsage:
        """Get token usage information for the most recent request.
        
        Returns:
            TokenUsage: Token usage information.
        """
        return self.last_usage


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider implementation.
    
    Uses the Google Gemini API to generate text responses.
    """
    
    # Gemini pricing per 1,000 tokens (as of 2025-04)
    # Source: https://ai.google.dev/pricing
    PRICING = {
        # Gemini models
        "gemini-pro": {"input": 0.00025, "output": 0.0005},  # Per 1K tokens
        "gemini-pro-vision": {"input": 0.00025, "output": 0.0005},
        "gemini-1.5-flash": {"input": 0.0003, "output": 0.0003},
        "gemini-1.5-pro": {"input": 0.0013, "output": 0.0013},
        "gemini-2.0-flash": {"input": 0.0001, "output": 0.0003},
        "gemini-2.0-flash-lite": {"input": 0.00005, "output": 0.00015},
    }
    
    def __init__(
        self,
        model_name: str,
        api_key: str = None,
        api_key_env_var: str = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the Gemini provider.
        
        Args:
            model_name: The name of the Gemini model to use (e.g., "gemini-1.5-flash").
            api_key: Direct API key for Gemini. If provided, takes precedence over api_key_env_var.
            api_key_env_var: The name of the environment variable containing the API key.
            temperature: Controls randomness in the output (0.0-1.0).
            max_tokens: The maximum number of tokens to generate.
            additional_params: Additional parameters to pass to the Gemini API.
            
        Raises:
            LLMError: If no API key is provided or found in the environment.
        """
        super().__init__()
        
        self.model_name = model_name
        
        # Get API key either directly or from environment variable
        if api_key:
            self.api_key = api_key
        elif api_key_env_var:
            self.api_key = os.environ.get(api_key_env_var)
        else:
            raise LLMError("Either api_key or api_key_env_var must be provided")
            
        if not self.api_key:
            raise LLMError(f"API key not found in environment variable {api_key_env_var}")
        
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.additional_params = additional_params or {}
        self.api_base = f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent"
        
        # Token usage for the most recent request
        self.last_usage = TokenUsage()
        
    def generate_text(self, prompt: str) -> str:
        """Generate text using the Google Gemini API.
        
        Args:
            prompt: The prompt to send to the Gemini API.
            
        Returns:
            str: The generated text response.
            
        Raises:
            LLMError: If there is an error generating text.
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
            response = requests.post(request_url, headers=headers, json=payload)
            response.raise_for_status()
            response_json = response.json()
            
            # Extract the generated text from the response
            try:
                generated_text = response_json["candidates"][0]["content"]["parts"][0]["text"]
                
                # Extract token usage information if available
                self._update_token_usage(response_json)
                
                return generated_text.strip()
            except (KeyError, IndexError) as e:
                raise LLMError(f"Error parsing Gemini API response structure: {str(e)}")
                
        except requests.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_message = e.response.json().get("error", {}).get("message", str(e))
                    raise LLMError(f"Error calling Gemini API: {error_message}")
                except (ValueError, AttributeError):
                    pass
            raise LLMError(f"Error calling Gemini API: {str(e)}")
            
    def _update_token_usage(self, response_json: Dict[str, Any]) -> None:
        """Update token usage information from the Gemini API response.
        
        Args:
            response_json: The JSON response from the Gemini API.
        """
        # Reset last usage
        self.last_usage = TokenUsage()
        
        # Extract token usage information if available
        if "usageMetadata" in response_json:
            usage = response_json["usageMetadata"]
            
            # Update last usage
            if "promptTokenCount" in usage:
                self.last_usage.prompt_tokens = usage["promptTokenCount"]
            
            if "candidatesTokenCount" in usage:
                self.last_usage.completion_tokens = usage["candidatesTokenCount"]
            
            # Calculate total tokens
            self.last_usage.total_tokens = self.last_usage.prompt_tokens + self.last_usage.completion_tokens
            
            # Calculate cost if pricing is available for this model
            model_key = self.model_name
            # Try to match based on model family if exact model not found
            if model_key not in self.PRICING:
                for pricing_model in self.PRICING:
                    if self.model_name.startswith(pricing_model):
                        model_key = pricing_model
                        break
            
            if model_key in self.PRICING:
                pricing = self.PRICING[model_key]
                prompt_cost = (self.last_usage.prompt_tokens / 1000) * pricing["input"]
                completion_cost = (self.last_usage.completion_tokens / 1000) * pricing["output"]
                self.last_usage.cost = prompt_cost + completion_cost
            
            # Update total usage
            self.total_usage.prompt_tokens += self.last_usage.prompt_tokens
            self.total_usage.completion_tokens += self.last_usage.completion_tokens
            self.total_usage.total_tokens += self.last_usage.total_tokens
            
            if self.last_usage.cost is not None:
                if self.total_usage.cost is None:
                    self.total_usage.cost = 0
                self.total_usage.cost += self.last_usage.cost
    
    def get_token_usage(self) -> TokenUsage:
        """Get token usage information for the most recent request.
        
        Returns:
            TokenUsage: Token usage information.
        """
        return self.last_usage
