"""Error handling utilities for LLM API calls.

Provides robust error handling with retry mechanisms, backoff strategies,
and categorized exceptions for LLM API interactions.
"""

from enum import Enum
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

# Set up logging
logger = logging.getLogger(__name__)

# Type variable for function return type
T = TypeVar('T')


class ErrorCategory(Enum):
    """Categories of errors that can occur during LLM API interactions."""
    
    # Authentication/Authorization errors
    AUTH = "authentication"
    
    # Rate limiting errors
    RATE_LIMIT = "rate_limit"
    
    # Network/connectivity errors
    NETWORK = "network"
    
    # Timeout errors
    TIMEOUT = "timeout"
    
    # Service unavailable errors
    SERVICE_UNAVAILABLE = "service_unavailable"
    
    # Invalid request errors
    INVALID_REQUEST = "invalid_request"
    
    # Content filtering/moderation errors
    CONTENT_FILTER = "content_filter"
    
    # Token context length errors
    CONTEXT_LENGTH = "context_length"
    
    # Response parsing errors
    PARSING = "parsing"
    
    # Unknown/other errors
    UNKNOWN = "unknown"


class LLMErrorBase(Exception):
    """Base class for all LLM-related exceptions."""
    
    def __init__(
        self, 
        message: str, 
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        status_code: Optional[int] = None,
        raw_error: Optional[Exception] = None,
        response: Optional[Any] = None,
        request_id: Optional[str] = None,
        retryable: bool = False
    ) -> None:
        """Initialize the LLM error.
        
        Args:
            message: Human-readable error message
            category: Category of error (auth, rate limit, etc.)
            status_code: HTTP status code if applicable
            raw_error: Original exception that caused this error
            response: Raw response data if available
            request_id: Request ID for tracking/debugging
            retryable: Whether this error can be retried
        """
        self.category = category
        self.status_code = status_code
        self.raw_error = raw_error
        self.response = response
        self.request_id = request_id
        self.retryable = retryable
        super().__init__(message)


class AuthError(LLMErrorBase):
    """Raised for authentication/authorization failures."""
    
    def __init__(
        self,
        message: str,
        **kwargs
    ) -> None:
        """Initialize the authentication error."""
        super().__init__(
            message,
            category=ErrorCategory.AUTH,
            retryable=False,  # Auth errors typically can't be retried without intervention
            **kwargs
        )


class RateLimitError(LLMErrorBase):
    """Raised when API rate limits are exceeded."""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        **kwargs
    ) -> None:
        """Initialize the rate limit error.
        
        Args:
            message: Human-readable error message
            retry_after: Seconds to wait before retrying (if provided by API)
            **kwargs: Additional arguments passed to the base class
        """
        self.retry_after = retry_after
        super().__init__(
            message,
            category=ErrorCategory.RATE_LIMIT,
            retryable=True,  # Rate limit errors are retryable after waiting
            **kwargs
        )


class NetworkError(LLMErrorBase):
    """Raised for network connectivity issues."""
    
    def __init__(
        self,
        message: str,
        **kwargs
    ) -> None:
        """Initialize the network error."""
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            retryable=True,  # Network errors are often transient
            **kwargs
        )


class TimeoutError(LLMErrorBase):
    """Raised when API requests time out."""
    
    def __init__(
        self,
        message: str,
        **kwargs
    ) -> None:
        """Initialize the timeout error."""
        super().__init__(
            message,
            category=ErrorCategory.TIMEOUT,
            retryable=True,  # Timeouts are often transient
            **kwargs
        )


class ServiceUnavailableError(LLMErrorBase):
    """Raised when the API service is unavailable."""
    
    def __init__(
        self,
        message: str,
        **kwargs
    ) -> None:
        """Initialize the service unavailable error."""
        super().__init__(
            message,
            category=ErrorCategory.SERVICE_UNAVAILABLE,
            retryable=True,  # Service unavailability is often temporary
            **kwargs
        )


class InvalidRequestError(LLMErrorBase):
    """Raised for invalid request errors (bad parameters, etc.)."""
    
    def __init__(
        self,
        message: str,
        **kwargs
    ) -> None:
        """Initialize the invalid request error."""
        super().__init__(
            message,
            category=ErrorCategory.INVALID_REQUEST,
            retryable=False,  # Invalid requests won't succeed without changes
            **kwargs
        )


class ContentFilterError(LLMErrorBase):
    """Raised when content is filtered/flagged by the LLM provider."""
    
    def __init__(
        self,
        message: str,
        **kwargs
    ) -> None:
        """Initialize the content filter error."""
        super().__init__(
            message,
            category=ErrorCategory.CONTENT_FILTER,
            retryable=False,  # Content filter issues won't be resolved by retrying
            **kwargs
        )


class ContextLengthError(LLMErrorBase):
    """Raised when the prompt or context length exceeds model limits."""
    
    def __init__(
        self,
        message: str,
        **kwargs
    ) -> None:
        """Initialize the context length error."""
        super().__init__(
            message,
            category=ErrorCategory.CONTEXT_LENGTH,
            retryable=False,  # Context length issues won't be resolved by retrying
            **kwargs
        )


class ParsingError(LLMErrorBase):
    """Raised for errors parsing API responses."""
    
    def __init__(
        self,
        message: str,
        **kwargs
    ) -> None:
        """Initialize the parsing error."""
        super().__init__(
            message,
            category=ErrorCategory.PARSING,
            retryable=False,  # Parsing errors generally won't be fixed by retrying
            **kwargs
        )


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        jitter_factor: float = 0.1,
        retry_error_types: Optional[List[Type[Exception]]] = None,
        retry_on_status_codes: Optional[List[int]] = None,
    ) -> None:
        """Initialize the retry configuration.
        
        Args:
            max_retries: Maximum number of retry attempts (0 means no retries)
            base_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            backoff_factor: Multiplier applied to delay between retries
            jitter: Whether to add randomness to the delay time
            jitter_factor: Factor to determine amount of jitter (0-1)
            retry_error_types: List of exception types to retry on
            retry_on_status_codes: List of HTTP status codes to retry on
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.jitter_factor = jitter_factor
        
        # Default error types that are generally safe to retry
        self.retry_error_types = retry_error_types or [
            NetworkError,
            TimeoutError,
            ServiceUnavailableError,
            RateLimitError
        ]
        
        # Default status codes that are generally safe to retry
        self.retry_on_status_codes = retry_on_status_codes or [
            408,  # Request Timeout
            429,  # Too Many Requests
            500,  # Internal Server Error
            502,  # Bad Gateway
            503,  # Service Unavailable
            504   # Gateway Timeout
        ]
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff and optional jitter.
        
        Args:
            attempt: Current retry attempt number (1-based)
            
        Returns:
            float: Delay time in seconds before next retry
        """
        # Exponential backoff: base_delay * (backoff_factor ^ (attempt - 1))
        delay = self.base_delay * (self.backoff_factor ** (attempt - 1))
        
        # Cap at max_delay
        delay = min(delay, self.max_delay)
        
        # Add jitter if enabled
        if self.jitter:
            # Add up to ±jitter_factor % of the delay as randomness
            import random
            jitter_amount = delay * self.jitter_factor
            delay += random.uniform(-jitter_amount, jitter_amount)
            
            # Ensure we don't go below base_delay/2
            delay = max(delay, self.base_delay / 2)
        
        return delay


def retry_with_backoff(
    func: Callable[..., T],
    retry_config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
    *args,
    **kwargs
) -> T:
    """Execute a function with retry logic and exponential backoff.
    
    Args:
        func: Function to execute
        retry_config: Retry configuration
        on_retry: Callback function called before each retry with:
                  (attempt_number, exception, next_delay)
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The return value of the function
        
    Raises:
        Exception: The last exception raised by the function if all retries fail
    """
    config = retry_config or RetryConfig()
    attempt = 0
    last_exception = None
    
    # Get function name safely (handle mocks in tests)
    func_name = getattr(func, "__name__", str(func))
    
    while attempt <= config.max_retries:
        try:
            # Attempt to execute the function
            return func(*args, **kwargs)
        
        except tuple(config.retry_error_types) as e:
            last_exception = e
            attempt += 1
            
            # Break if we've hit the retry limit
            if attempt > config.max_retries:
                break
                
            # Check if the error type suggests a specific retry delay
            retry_after = getattr(e, 'retry_after', None)
            
            # Calculate delay
            if retry_after is not None:
                delay = retry_after
            else:
                delay = config.calculate_delay(attempt)
                
            # Log the retry
            logger.warning(
                f"Retry {attempt}/{config.max_retries} for {func_name} "
                f"after {delay:.2f}s due to {e.__class__.__name__}: {str(e)}"
            )
            
            # Call the retry callback if provided
            if on_retry:
                try:
                    on_retry(attempt, e, delay)
                except Exception as callback_error:
                    logger.error(f"Error in retry callback: {callback_error}")
            
            # Wait before the next retry
            time.sleep(delay)
        
        except Exception as e:
            # Don't retry other exceptions
            last_exception = e
            break
    
    # If we've exhausted our retries or encountered a non-retryable exception
    if last_exception:
        logger.error(
            f"All {attempt} retries failed for {func_name}. "
            f"Last error: {last_exception.__class__.__name__}: {str(last_exception)}"
        )
        raise last_exception


def classify_error(
    error: Exception,
    response: Any = None,
    status_code: Optional[int] = None
) -> Type[LLMErrorBase]:
    """Classify an error into an appropriate LLM error type.
    
    Args:
        error: The original exception
        response: The API response, if available
        status_code: The HTTP status code, if available
        
    Returns:
        Type[LLMErrorBase]: The appropriate error class
    """
    # Default error details
    error_message = str(error)
    
    # Determine error category based on status code and response content
    if status_code is not None:
        if status_code == 401 or status_code == 403:
            return AuthError
        elif status_code == 429:
            return RateLimitError
        elif status_code == 408 or status_code == 504:
            return TimeoutError
        elif status_code in (500, 502, 503):
            return ServiceUnavailableError
        elif 400 <= status_code < 500:
            return InvalidRequestError
    
    # Try to determine error type from the error name/message for common providers
    error_type = error.__class__.__name__.lower()
    error_str = error_message.lower()
    
    # Order matters here - more specific patterns should come first
    
    # Check for parsing errors first since keywords like "invalid" might overlap with other categories
    if any(kw in error_type or kw in error_str for kw in ("parse", "parsing", "json")):
        return ParsingError
        
    if any(kw in error_type or kw in error_str for kw in ("api key", "auth", "unauthorized", "forbidden", "apikey")):
        return AuthError
    elif any(kw in error_type or kw in error_str for kw in ("rate", "ratelimit", "quota", "limit", "too many")) and "token limit" not in error_str:
        return RateLimitError
    elif any(kw in error_type or kw in error_str for kw in ("network", "connection", "connecterror")):
        return NetworkError
    elif any(kw in error_type or kw in error_str for kw in ("timeout", "timed out")):
        return TimeoutError
    elif any(kw in error_type or kw in error_str for kw in ("unavailable", "down", "maintenance", "overloaded")):
        return ServiceUnavailableError
    elif any(kw in error_type or kw in error_str for kw in ("invalid", "validation", "parameter", "badrequest", "bad request")):
        return InvalidRequestError
    elif any(kw in error_type or kw in error_str for kw in ("content", "filter", "moderation", "abuse", "blocked", "policy")):
        return ContentFilterError
    elif any(kw in error_type or kw in error_str for kw in ("context", "length", "token", "too long", "truncated", "overflow")):
        return ContextLengthError
    elif any(kw in error_type or kw in error_str for kw in ("syntax", "format")):
        return ParsingError
    
    # Default error type if we couldn't classify it
    return LLMErrorBase