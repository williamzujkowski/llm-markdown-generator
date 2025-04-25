#!/usr/bin/env python3
"""
Integration test to verify OpenAI and Gemini API keys.

This script verifies that the API keys in your .env file are working
by making a minimal API call to each provider. It's useful for quick
verification of API connectivity and key validity.
"""

import os
import json
import requests
from dotenv import load_dotenv

# Update the import path to access the main package
from llm_markdown_generator.llm_provider import OpenAIProvider, GeminiProvider
from llm_markdown_generator.error_handler import (
    AuthError,
    NetworkError,
    LLMErrorBase,
    RetryConfig
)

def test_openai_api_key():
    """Test the OpenAI API key with a minimal request."""
    try:
        # Create OpenAI provider using the OPENAI_API_KEY environment variable
        provider = OpenAIProvider(
            model_name="gpt-3.5-turbo",  # Using a smaller model for the test
            api_key_env_var="OPENAI_API_KEY",
            temperature=0.7,
            max_tokens=10,  # Minimal token generation for testing
            retry_config=RetryConfig(max_retries=0)  # Disable retries for testing
        )
        
        # Make a simple API call
        response = provider.generate_text("Say hello:")
        
        print(f"✅ OpenAI API key is working!")
        print(f"   Response: {response}")
        return True
    
    except AuthError as e:
        print(f"❌ OpenAI API key test failed: {str(e)}")
        return False
    
    except NetworkError as e:
        print(f"❌ Network error testing OpenAI API: {str(e)}")
        return False
    
    except LLMErrorBase as e:
        print(f"❌ Error testing OpenAI API key: {str(e)}")
        return False
    
    except Exception as e:
        print(f"❌ Unexpected error testing OpenAI API key: {str(e)}")
        return False

def test_gemini_api_key():
    """Test the Gemini API key with the provider class."""
    try:
        # Get API key from environment
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        # Create Gemini provider
        provider = GeminiProvider(
            model_name="gemini-2.0-flash",  # Use the Gemini 2.0 Flash model
            api_key=api_key,
            temperature=0.7,
            max_tokens=10,  # Minimal token generation for testing
            retry_config=RetryConfig(max_retries=0)  # Disable retries for testing
        )
        
        # Make a simple API call
        response = provider.generate_text("Say hello:")
        
        print(f"✅ Gemini API key is working!")
        print(f"   Response: {response}")
        return True
    
    except AuthError as e:
        print(f"❌ Gemini API key test failed: {str(e)}")
        return False
    
    except NetworkError as e:
        print(f"❌ Network error testing Gemini API: {str(e)}")
        return False
    
    except LLMErrorBase as e:
        print(f"❌ Error testing Gemini API key: {str(e)}")
        return False
    
    except Exception as e:
        print(f"❌ Unexpected error testing Gemini API key: {str(e)}")
        return False

def main():
    """Main function to test both API keys."""
    print("Testing LLM API keys...\n")
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if environment variables are set
    openai_key = os.environ.get("OPENAI_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    
    if not openai_key:
        print("❌ OPENAI_API_KEY environment variable is not set in .env file")
    
    if not gemini_key:
        print("❌ GEMINI_API_KEY environment variable is not set in .env file")
    
    if not openai_key and not gemini_key:
        print("\nPlease create a .env file with your API keys:")
        print("OPENAI_API_KEY=your_openai_api_key_here")
        print("GEMINI_API_KEY=your_gemini_api_key_here")
        return
    
    print("Environment variables loaded, testing API connections...\n")
    
    # Test OpenAI API key if set
    if openai_key:
        openai_success = test_openai_api_key()
        print("")
    else:
        openai_success = False
    
    # Test Gemini API key if set
    if gemini_key:
        gemini_success = test_gemini_api_key()
        print("")
    else:
        gemini_success = False
    
    # Summary
    print("=== API Key Test Summary ===")
    if openai_key:
        status = "✅ WORKING" if openai_success else "❌ FAILED"
        print(f"OpenAI API Key: {status}")
    else:
        print("OpenAI API Key: ❌ NOT CONFIGURED")
    
    if gemini_key:
        status = "✅ WORKING" if gemini_success else "❌ FAILED"
        print(f"Gemini API Key: {status}")
    else:
        print("Gemini API Key: ❌ NOT CONFIGURED")

if __name__ == "__main__":
    main()