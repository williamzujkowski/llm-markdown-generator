#!/usr/bin/env python
"""
Demo script showing how to use the robust error handling system.

This script demonstrates:
1. Creating custom retry configurations
2. Handling different error types
3. Using error attributes for detailed information
"""

import os
import sys
import logging
from pathlib import Path

# Add the parent directory to the path to import the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from llm_markdown_generator.error_handler import (
    AuthError,
    ContentFilterError,
    ContextLengthError,
    InvalidRequestError,
    NetworkError,
    ParsingError,
    RateLimitError,
    RetryConfig,
    ServiceUnavailableError,
    TimeoutError
)
from llm_markdown_generator.llm_provider import OpenAIProvider, GeminiProvider

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def demonstrate_retry_config():
    """Demonstrate custom retry configuration."""
    # Default retry config uses 3 retries, 1.0s base delay, x2 backoff
    default_config = RetryConfig()
    
    # Custom aggressive retry config for intermittent network issues
    network_config = RetryConfig(
        max_retries=5,  # More retries
        base_delay=0.5, # Start with shorter delay
        max_delay=30.0, # Cap maximum delay
        backoff_factor=1.5, # More moderate backoff
    )
    
    # Custom retry config for rate limiting (longer delays)
    rate_limit_config = RetryConfig(
        max_retries=3,
        base_delay=2.0,     # Start with longer delay
        backoff_factor=3.0, # More aggressive backoff
        jitter=True,        # Add randomness to prevent thundering herd
        # Only retry on rate limit errors
        retry_error_types=[RateLimitError],
    )
    
    # Log configurations
    logger.info(f"Default config: max_retries={default_config.max_retries}, " 
                f"base_delay={default_config.base_delay}s")
    logger.info(f"Network config: max_retries={network_config.max_retries}, "
                f"base_delay={network_config.base_delay}s")
    logger.info(f"Rate limit config: max_retries={rate_limit_config.max_retries}, "
                f"base_delay={rate_limit_config.base_delay}s")
    
    # Example delay sequence for each config
    logger.info("\nExample delay sequences for 4 retries:")
    
    logger.info("Default config:")
    for i in range(1, 5):
        logger.info(f"  Retry {i}: {default_config.calculate_delay(i):.2f}s")
    
    logger.info("Network config:")
    for i in range(1, 6):
        logger.info(f"  Retry {i}: {network_config.calculate_delay(i):.2f}s")
    
    logger.info("Rate limit config:")
    for i in range(1, 4):
        logger.info(f"  Retry {i}: {rate_limit_config.calculate_delay(i):.2f}s")


def generate_text_with_safety(provider_type="openai", prompt=""):
    """Generate text with error handling.
    
    Args:
        provider_type: The LLM provider to use ("openai" or "gemini")
        prompt: The text prompt (if empty, a default prompt is used)
    """
    # Configure a retry strategy
    retry_config = RetryConfig(
        max_retries=2,
        base_delay=1.0,
        jitter=True
    )
    
    # Use a default prompt if none provided
    if not prompt:
        prompt = "Write a paragraph about error handling in software development."
    
    try:
        # Create the appropriate provider
        if provider_type.lower() == "openai":
            if "OPENAI_API_KEY" not in os.environ:
                raise AuthError("OPENAI_API_KEY environment variable not set")
            
            provider = OpenAIProvider(
                model_name="gpt-4o-mini",  # Using a smaller model
                api_key_env_var="OPENAI_API_KEY",
                temperature=0.7,
                retry_config=retry_config
            )
            logger.info(f"Using OpenAI provider with model: gpt-4o-mini")
        elif provider_type.lower() == "gemini":
            if "GEMINI_API_KEY" not in os.environ:
                raise AuthError("GEMINI_API_KEY environment variable not set")
            
            provider = GeminiProvider(
                model_name="gemini-2.0-flash",
                api_key_env_var="GEMINI_API_KEY",
                temperature=0.7,
                retry_config=retry_config
            )
            logger.info(f"Using Gemini provider with model: gemini-2.0-flash")
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")
        
        # Generate text
        logger.info(f"Generating text with prompt: {prompt}")
        response = provider.generate_text(prompt)
        
        # Log result
        logger.info(f"Generated {len(response)} characters")
        
        return response
    
    except AuthError as e:
        logger.error(f"Authentication error: {e}")
        logger.error("Please check your API key in the environment variables.")
    
    except RateLimitError as e:
        retry_msg = f" Retry after: {e.retry_after}s" if e.retry_after else ""
        logger.error(f"Rate limit exceeded: {e}{retry_msg}")
        
    except NetworkError as e:
        logger.error(f"Network error: {e}")
        logger.error("Please check your internet connection.")
        
    except TimeoutError as e:
        logger.error(f"Request timed out: {e}")
        
    except ServiceUnavailableError as e:
        logger.error(f"Service unavailable: {e}")
        logger.error("The LLM service is temporarily unavailable.")
        
    except ContentFilterError as e:
        logger.error(f"Content filtered: {e}")
        logger.error("The prompt or response was flagged by the content filter.")
        
    except ContextLengthError as e:
        logger.error(f"Context length exceeded: {e}")
        logger.error("Try shortening your prompt or using a model with larger context.")
        
    except InvalidRequestError as e:
        logger.error(f"Invalid request: {e}")
        logger.error("Check the parameters of your request.")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    
    return None


def main():
    """Main function to demonstrate error handling."""
    logger.info("======= LLM Error Handling Demo =======")
    
    # Demonstrate retry configurations
    logger.info("\n--- Retry Configuration Examples ---")
    demonstrate_retry_config()
    
    # Generate text with proper error handling
    logger.info("\n--- Text Generation with Error Handling ---")
    
    # Try to use the provider specified in command line args or default to OpenAI
    provider_type = sys.argv[1] if len(sys.argv) > 1 else "openai"
    
    response = generate_text_with_safety(provider_type)
    if response:
        print("\nGenerated text:")
        print(response)
    

if __name__ == "__main__":
    main()