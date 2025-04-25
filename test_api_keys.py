#!/usr/bin/env python3
"""
Simple script to test OpenAI and Gemini API keys.

This script verifies that the API keys in your .env file are working
by making a minimal API call to each provider.
"""

import os
import json
import requests
from dotenv import load_dotenv
from src.llm_markdown_generator.llm_provider import OpenAIProvider, LLMError

def test_openai_api_key():
    """Test the OpenAI API key with a minimal request."""
    try:
        # Create OpenAI provider using the OPENAI_API_KEY environment variable
        provider = OpenAIProvider(
            model_name="gpt-3.5-turbo",  # Using a smaller model for the test
            api_key_env_var="OPENAI_API_KEY",
            temperature=0.7,
            max_tokens=10  # Minimal token generation for testing
        )
        
        # Make a simple API call
        response = provider.generate_text("Say hello:")
        
        print(f"✅ OpenAI API key is working!")
        print(f"   Response: {response}")
        return True
    
    except LLMError as e:
        print(f"❌ OpenAI API key test failed: {str(e)}")
        return False
    
    except Exception as e:
        print(f"❌ Unexpected error testing OpenAI API key: {str(e)}")
        return False

def test_gemini_api_key():
    """Test the Gemini API key with a minimal direct API request."""
    try:
        # Get API key from environment
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        # Make a direct API call to Gemini
        model = "gemini-2.0-flash"  # Use the Gemini 2.0 Flash model
        url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"
        
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": "Say hello:"
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 10
            }
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()
        
        # Extract the generated text
        try:
            generated_text = response_json["candidates"][0]["content"]["parts"][0]["text"]
            print(f"✅ Gemini API key is working!")
            print(f"   Response: {generated_text.strip()}")
            return True
        except (KeyError, IndexError) as e:
            print(f"❌ Error parsing Gemini API response: {str(e)}")
            print(f"   Raw response: {json.dumps(response_json, indent=2)}")
            return False
        
    except requests.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_message = e.response.json().get("error", {}).get("message", str(e))
                print(f"❌ Gemini API key test failed: {error_message}")
            except (ValueError, AttributeError):
                print(f"❌ Gemini API key test failed: {str(e)}")
        else:
            print(f"❌ Gemini API key test failed: {str(e)}")
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