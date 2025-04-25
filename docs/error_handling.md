# Advanced Error Handling

This document describes the advanced error handling system implemented in the LLM Markdown Generator.

## Overview

The LLM Markdown Generator includes a robust error handling system that provides:

1. **Categorized exceptions** - Different exception types for different error categories
2. **Detailed error information** - Rich error details including status codes, responses, and causes
3. **Automatic retries** - Configurable retry mechanism with exponential backoff
4. **Smart error classification** - Intelligent classification of errors based on status codes and messages

## Error Categories

The system defines several categories of errors that can occur when interacting with LLM providers:

| Category | Description | Retryable? |
|----------|-------------|------------|
| Authentication | API key or authentication issues | No |
| Rate Limit | API rate limits exceeded | Yes |
| Network | Network connectivity issues | Yes |
| Timeout | Request timeout issues | Yes |
| Service Unavailable | LLM service unavailability | Yes |
| Invalid Request | Malformed requests or parameters | No |
| Content Filter | Content filtered by safety systems | No |
| Context Length | Token/context length exceeded | No |
| Parsing | Response parsing errors | No |
| Unknown | Unclassified errors | No |

## Exception Hierarchy

All errors extend from a base `LLMErrorBase` exception class:

```
LLMErrorBase
├── AuthError
├── RateLimitError
├── NetworkError
├── TimeoutError
├── ServiceUnavailableError
├── InvalidRequestError
├── ContentFilterError
├── ContextLengthError
└── ParsingError
```

Each exception provides detailed information about the error:

- Human-readable error message
- Error category
- HTTP status code (if applicable)
- Original exception (if any)
- Raw API response (if available)
- Request ID for tracking (if provided by the API)
- Retryability flag

## Retry Mechanism

The system includes a configurable retry mechanism with exponential backoff:

```python
from llm_markdown_generator.error_handler import RetryConfig, retry_with_backoff

# Create a custom retry configuration
retry_config = RetryConfig(
    max_retries=3,               # Maximum number of retry attempts
    base_delay=1.0,              # Initial delay in seconds
    max_delay=60.0,              # Maximum delay between retries
    backoff_factor=2.0,          # Multiplier applied to delay between retries
    jitter=True,                 # Add randomness to retry timing
    jitter_factor=0.1,           # Factor to determine jitter amount (0-1)
    retry_error_types=[...],     # List of error types to retry
    retry_on_status_codes=[...]  # List of HTTP status codes to retry
)

# Use retry with a function
result = retry_with_backoff(
    my_function,                 # Function to call with retry logic
    retry_config=retry_config,   # Retry configuration (optional)
    on_retry=my_callback,        # Callback before each retry (optional)
    *args, **kwargs              # Arguments to pass to the function
)
```

By default, the system will retry on network errors, timeouts, service unavailability, and rate limit errors.

## Automatic Error Classification

The system can automatically classify errors based on status codes and error messages:

```python
from llm_markdown_generator.error_handler import classify_error

# Classify an exception
error_type = classify_error(exception, response_data, status_code)

# Raise the appropriate error type
raise error_type("Error message", status_code=status_code, response=response_data)
```

## Usage in LLM Providers

Both OpenAI and Gemini providers automatically use the error handling system:

```python
from llm_markdown_generator.llm_provider import OpenAIProvider
from llm_markdown_generator.error_handler import RetryConfig, AuthError, RateLimitError

# Configure retries
retry_config = RetryConfig(max_retries=5, base_delay=2.0)

try:
    provider = OpenAIProvider(
        model_name="gpt-4o",
        api_key_env_var="OPENAI_API_KEY",
        retry_config=retry_config
    )
    result = provider.generate_text("My prompt")
    
except AuthError as e:
    print(f"Authentication error: {e}")
    
except RateLimitError as e:
    print(f"Rate limit exceeded. Suggested retry after: {e.retry_after} seconds")
    
except Exception as e:
    print(f"Other error: {e}")
```

## Logging

The error handling system integrates with Python's logging system:

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Logger automatically captures retry attempts, errors, and other information
```

## Best Practices

1. **Catch specific exceptions** - Catch the most specific exception types that make sense for your application
2. **Use retry with caution** - Configure retry parameters appropriately for your use case
3. **Monitor retries** - Use the logging system to track retry patterns and adjust accordingly
4. **Configure timeouts** - Set reasonable timeouts for API requests
5. **Handle rate limits** - Pay attention to rate limit errors and implement rate limiting on your side if needed