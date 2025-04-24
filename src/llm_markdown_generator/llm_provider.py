"""LLM Provider interfaces and implementations.

Defines the abstract base class for LLM providers and implements
specific provider clients (OpenAI and Google Gemini).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import json
import os
import requests


class LLMError(Exception):
    """Raised when there is an error interacting with an LLM provider."""

    pass


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    This defines the interface that all LLM provider implementations must follow.
    """

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
        self.model_name = model_name
        self.api_key = os.environ.get(api_key_env_var)
        if not self.api_key:
            raise LLMError(f"API key not found in environment variable {api_key_env_var}")

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.additional_params = additional_params or {}
        self.api_base = "https://api.openai.com/v1/chat/completions"

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
            return generated_text.strip()

        except requests.RequestException as e:
            raise LLMError(f"Error calling OpenAI API: {str(e)}")
        except (KeyError, IndexError) as e:
            raise LLMError(f"Error parsing OpenAI API response: {str(e)}")


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
    ) -> None:
        """Initialize the Gemini provider.
        
        Args:
            model_name: The name of the Gemini model to use (e.g., "gemini-1.0-pro").
            api_key: Direct API key for Gemini. If provided, takes precedence over api_key_env_var.
            api_key_env_var: The name of the environment variable containing the API key.
            temperature: Controls randomness in the output (0.0-1.0).
            max_tokens: The maximum number of tokens to generate.
            additional_params: Additional parameters to pass to the Gemini API.
            
        Raises:
            LLMError: If no API key is provided or found in the environment.
        """
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
