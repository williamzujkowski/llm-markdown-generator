"""Token usage tracking utilities.

This module provides utilities for tracking and reporting token usage
across different LLM providers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Union
import json
import os
from pathlib import Path

from llm_markdown_generator.llm_provider import TokenUsage


@dataclass
class UsageRecord:
    """A record of token usage for a specific operation.

    Attributes:
        timestamp: When the operation occurred.
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        total_tokens: Total tokens used.
        cost: Estimated cost in USD.
        provider: The LLM provider used.
        model: The model used.
        operation: The type of operation (e.g., "generate_content").
        topic: The topic or subject of the operation, if applicable.
        metadata: Additional metadata about the operation.
    """
    timestamp: datetime = field(default_factory=datetime.now)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: Optional[float] = None
    provider: str = "unknown"
    model: str = "unknown"
    operation: str = "unknown"
    topic: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert the record to a dictionary.

        Returns:
            Dict: The record as a dictionary.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "provider": self.provider,
            "model": self.model,
            "operation": self.operation,
            "topic": self.topic,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "UsageRecord":
        """Create a UsageRecord from a dictionary.

        Args:
            data: The dictionary to create the record from.

        Returns:
            UsageRecord: The created record.
        """
        # Parse timestamp if it's a string
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        return cls(
            timestamp=timestamp,
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            cost=data.get("cost"),
            provider=data.get("provider", "unknown"),
            model=data.get("model", "unknown"),
            operation=data.get("operation", "unknown"),
            topic=data.get("topic"),
            metadata=data.get("metadata", {})
        )


class TokenTracker:
    """Tracks token usage across operations and provides reporting functionality.

    This class maintains a history of token usage and can provide summaries
    and reports. It can also persist usage data to a file for later analysis.
    """

    def __init__(self, log_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize the token tracker.

        Args:
            log_path: Optional path to a file where usage records will be logged.
                If not provided, usage will only be tracked in memory.
        """
        self.records: List[UsageRecord] = []
        self.total_usage = TokenUsage()
        self.log_path = Path(log_path) if log_path else None

        # Load existing records if log file exists
        if self.log_path and self.log_path.exists():
            self._load_records()

    def record_usage(
        self,
        token_usage: TokenUsage,
        provider: str,
        model: str,
        operation: str,
        topic: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> UsageRecord:
        """Record token usage for an operation.

        Args:
            token_usage: The token usage to record.
            provider: The LLM provider used.
            model: The model used.
            operation: The type of operation.
            topic: The topic or subject, if applicable.
            metadata: Additional metadata about the operation.

        Returns:
            UsageRecord: The created usage record.
        """
        # Create a record
        record = UsageRecord(
            timestamp=datetime.now(),
            prompt_tokens=token_usage.prompt_tokens,
            completion_tokens=token_usage.completion_tokens,
            total_tokens=token_usage.total_tokens,
            cost=token_usage.cost,
            provider=provider,
            model=model,
            operation=operation,
            topic=topic,
            metadata=metadata or {}
        )

        # Add to history
        self.records.append(record)

        # Update total usage
        self.total_usage.prompt_tokens += token_usage.prompt_tokens
        self.total_usage.completion_tokens += token_usage.completion_tokens
        self.total_usage.total_tokens += token_usage.total_tokens

        if token_usage.cost is not None:
            if self.total_usage.cost is None:
                self.total_usage.cost = 0
            self.total_usage.cost += token_usage.cost

        # Save to file if configured
        if self.log_path:
            self._append_record_to_log(record)

        return record

    def get_total_usage(self) -> TokenUsage:
        """Get the total token usage across all records.

        Returns:
            TokenUsage: The total token usage.
        """
        return self.total_usage

    def get_usage_by_provider(self) -> Dict[str, TokenUsage]:
        """Get token usage grouped by provider.

        Returns:
            Dict[str, TokenUsage]: Token usage for each provider.
        """
        provider_usage: Dict[str, TokenUsage] = {}

        for record in self.records:
            if record.provider not in provider_usage:
                provider_usage[record.provider] = TokenUsage()

            usage = provider_usage[record.provider]
            usage.prompt_tokens += record.prompt_tokens
            usage.completion_tokens += record.completion_tokens
            usage.total_tokens += record.total_tokens

            if record.cost is not None:
                if usage.cost is None:
                    usage.cost = 0
                usage.cost += record.cost

        return provider_usage

    def get_usage_by_model(self) -> Dict[str, TokenUsage]:
        """Get token usage grouped by model.

        Returns:
            Dict[str, TokenUsage]: Token usage for each model.
        """
        model_usage: Dict[str, TokenUsage] = {}

        for record in self.records:
            if record.model not in model_usage:
                model_usage[record.model] = TokenUsage()

            usage = model_usage[record.model]
            usage.prompt_tokens += record.prompt_tokens
            usage.completion_tokens += record.completion_tokens
            usage.total_tokens += record.total_tokens

            if record.cost is not None:
                if usage.cost is None:
                    usage.cost = 0
                usage.cost += record.cost

        return model_usage

    def get_usage_by_topic(self) -> Dict[str, TokenUsage]:
        """Get token usage grouped by topic.

        Returns:
            Dict[str, TokenUsage]: Token usage for each topic.
        """
        topic_usage: Dict[str, TokenUsage] = {}

        for record in self.records:
            # Skip records without a topic
            if not record.topic:
                continue

            if record.topic not in topic_usage:
                topic_usage[record.topic] = TokenUsage()

            usage = topic_usage[record.topic]
            usage.prompt_tokens += record.prompt_tokens
            usage.completion_tokens += record.completion_tokens
            usage.total_tokens += record.total_tokens

            if record.cost is not None:
                if usage.cost is None:
                    usage.cost = 0
                usage.cost += record.cost

        return topic_usage

    def get_usage_summary(self) -> Dict:
        """Get a summary of token usage statistics.

        Returns:
            Dict: A summary of token usage.
        """
        # Total usage
        total_usage = self.get_total_usage()

        # Usage by provider
        provider_usage = self.get_usage_by_provider()

        # Usage by model
        model_usage = self.get_usage_by_model()

        # Count of operations
        operations = {}
        for record in self.records:
            operations[record.operation] = operations.get(record.operation, 0) + 1

        # Cost summary
        total_cost = total_usage.cost or 0
        cost_by_provider = {
            provider: usage.cost or 0
            for provider, usage in provider_usage.items()
        }

        return {
            "total_tokens": total_usage.total_tokens,
            "prompt_tokens": total_usage.prompt_tokens,
            "completion_tokens": total_usage.completion_tokens,
            "total_cost": total_cost,
            "operation_count": len(self.records),
            "operations": operations,
            "cost_by_provider": cost_by_provider,
            "providers": list(provider_usage.keys()),
            "models": list(model_usage.keys()),
        }

    def generate_report(self, detailed: bool = False) -> str:
        """Generate a formatted report of token usage.

        Args:
            detailed: Whether to include detailed usage information.

        Returns:
            str: A formatted report.
        """
        summary = self.get_usage_summary()
        provider_usage = self.get_usage_by_provider()
        model_usage = self.get_usage_by_model()

        # Format the report
        report = [
            "Token Usage Report",
            "=================",
            f"Total Operations: {summary['operation_count']}",
            f"Total Tokens: {summary['total_tokens']:,}",
            f"  - Prompt Tokens: {summary['prompt_tokens']:,}",
            f"  - Completion Tokens: {summary['completion_tokens']:,}",
            f"Estimated Total Cost: ${summary['total_cost']:.6f}"
        ]

        # Add provider breakdown
        report.append("\nUsage by Provider:")
        for provider, usage in provider_usage.items():
            cost_str = f", Cost: ${usage.cost:.6f}" if usage.cost is not None else ""
            report.append(
                f"  - {provider}: {usage.total_tokens:,} tokens{cost_str}"
            )

        # Add model breakdown
        report.append("\nUsage by Model:")
        for model, usage in model_usage.items():
            cost_str = f", Cost: ${usage.cost:.6f}" if usage.cost is not None else ""
            report.append(
                f"  - {model}: {usage.total_tokens:,} tokens{cost_str}"
            )

        # Add operation counts
        report.append("\nOperations:")
        for operation, count in summary["operations"].items():
            report.append(f"  - {operation}: {count} calls")

        # Add detailed records if requested
        if detailed and self.records:
            report.append("\nDetailed Records:")
            for i, record in enumerate(self.records[-10:]):  # Show last 10 records
                cost_str = f", Cost: ${record.cost:.6f}" if record.cost is not None else ""
                report.append(
                    f"  {i+1}. [{record.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"{record.operation} - {record.provider}/{record.model}: "
                    f"{record.total_tokens} tokens{cost_str}"
                )

            if len(self.records) > 10:
                report.append(f"  ... and {len(self.records) - 10} more records")

        return "\n".join(report)

    def _append_record_to_log(self, record: UsageRecord) -> None:
        """Append a usage record to the log file.

        Args:
            record: The record to append.
        """
        if not self.log_path:
            return

        # Ensure the directory exists
        os.makedirs(self.log_path.parent, exist_ok=True)

        # Write the record as a JSON line
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def _load_records(self) -> None:
        """Load usage records from the log file."""
        if not self.log_path or not self.log_path.exists():
            return

        try:
            with open(self.log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record_data = json.loads(line)
                        record = UsageRecord.from_dict(record_data)
                        self.records.append(record)
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue

            # Recalculate total usage
            total = TokenUsage()
            for record in self.records:
                total.prompt_tokens += record.prompt_tokens
                total.completion_tokens += record.completion_tokens
                total.total_tokens += record.total_tokens

                if record.cost is not None:
                    if total.cost is None:
                        total.cost = 0
                    total.cost += record.cost

            self.total_usage = total
        except Exception:
            # If there's any error loading the records, just start with an empty list
            self.records = []
            self.total_usage = TokenUsage()