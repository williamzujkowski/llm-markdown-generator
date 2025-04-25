"""Unit tests for the token tracker.

Tests the token tracking and reporting functionality.
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest import mock

import pytest

from llm_markdown_generator.llm_provider import TokenUsage
from llm_markdown_generator.token_tracker import TokenTracker, UsageRecord


class TestUsageRecord:
    """Tests for the UsageRecord class."""

    def test_create_usage_record(self):
        """Test creating a usage record."""
        record = UsageRecord(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            cost=0.001,
            provider="openai",
            model="gpt-3.5-turbo",
            operation="generate_content",
            topic="test_topic",
            metadata={"test_key": "test_value"}
        )

        assert record.prompt_tokens == 10
        assert record.completion_tokens == 20
        assert record.total_tokens == 30
        assert record.cost == 0.001
        assert record.provider == "openai"
        assert record.model == "gpt-3.5-turbo"
        assert record.operation == "generate_content"
        assert record.topic == "test_topic"
        assert record.metadata == {"test_key": "test_value"}

    def test_to_dict(self):
        """Test converting a record to a dictionary."""
        test_time = datetime(2025, 1, 1, 12, 0, 0)
        record = UsageRecord(
            timestamp=test_time,
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            cost=0.001,
            provider="openai",
            model="gpt-3.5-turbo",
            operation="generate_content",
            topic="test_topic",
            metadata={"test_key": "test_value"}
        )

        record_dict = record.to_dict()
        assert record_dict["timestamp"] == test_time.isoformat()
        assert record_dict["prompt_tokens"] == 10
        assert record_dict["completion_tokens"] == 20
        assert record_dict["total_tokens"] == 30
        assert record_dict["cost"] == 0.001
        assert record_dict["provider"] == "openai"
        assert record_dict["model"] == "gpt-3.5-turbo"
        assert record_dict["operation"] == "generate_content"
        assert record_dict["topic"] == "test_topic"
        assert record_dict["metadata"] == {"test_key": "test_value"}

    def test_from_dict(self):
        """Test creating a record from a dictionary."""
        test_time = datetime(2025, 1, 1, 12, 0, 0)
        record_dict = {
            "timestamp": test_time.isoformat(),
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "cost": 0.001,
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "operation": "generate_content",
            "topic": "test_topic",
            "metadata": {"test_key": "test_value"}
        }

        record = UsageRecord.from_dict(record_dict)
        assert record.timestamp == test_time
        assert record.prompt_tokens == 10
        assert record.completion_tokens == 20
        assert record.total_tokens == 30
        assert record.cost == 0.001
        assert record.provider == "openai"
        assert record.model == "gpt-3.5-turbo"
        assert record.operation == "generate_content"
        assert record.topic == "test_topic"
        assert record.metadata == {"test_key": "test_value"}

    def test_from_dict_with_missing_fields(self):
        """Test creating a record from a dictionary with missing fields."""
        # Missing several fields
        record_dict = {
            "prompt_tokens": 10,
            "completion_tokens": 20
        }

        record = UsageRecord.from_dict(record_dict)
        assert record.prompt_tokens == 10
        assert record.completion_tokens == 20
        assert record.total_tokens == 0  # Default value
        assert record.cost is None  # Default value
        assert record.provider == "unknown"  # Default value
        assert record.model == "unknown"  # Default value
        assert record.operation == "unknown"  # Default value
        assert record.topic is None  # Default value
        assert record.metadata == {}  # Default value


class TestTokenTracker:
    """Tests for the TokenTracker class."""

    def test_record_usage(self):
        """Test recording token usage."""
        tracker = TokenTracker()
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30, cost=0.001)

        record = tracker.record_usage(
            token_usage=usage,
            provider="openai",
            model="gpt-3.5-turbo",
            operation="generate_content",
            topic="test_topic",
            metadata={"test_key": "test_value"}
        )

        # Check the record
        assert record.prompt_tokens == 10
        assert record.completion_tokens == 20
        assert record.total_tokens == 30
        assert record.cost == 0.001
        assert record.provider == "openai"
        assert record.model == "gpt-3.5-turbo"
        assert record.operation == "generate_content"
        assert record.topic == "test_topic"
        assert record.metadata == {"test_key": "test_value"}

        # Check that the record was added to the tracker
        assert len(tracker.records) == 1
        assert tracker.records[0] == record

        # Check that the total usage was updated
        assert tracker.total_usage.prompt_tokens == 10
        assert tracker.total_usage.completion_tokens == 20
        assert tracker.total_usage.total_tokens == 30
        assert tracker.total_usage.cost == 0.001

    def test_multiple_records(self):
        """Test recording multiple token usage records."""
        tracker = TokenTracker()

        # Add two usage records
        usage1 = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30, cost=0.001)
        usage2 = TokenUsage(prompt_tokens=15, completion_tokens=25, total_tokens=40, cost=0.002)

        tracker.record_usage(
            token_usage=usage1,
            provider="openai",
            model="gpt-3.5-turbo",
            operation="generate_content",
            topic="topic1"
        )

        tracker.record_usage(
            token_usage=usage2,
            provider="gemini",
            model="gemini-1.5-flash",
            operation="generate_content",
            topic="topic2"
        )

        # Check that both records were added
        assert len(tracker.records) == 2

        # Check that the total usage was updated correctly
        assert tracker.total_usage.prompt_tokens == 25
        assert tracker.total_usage.completion_tokens == 45
        assert tracker.total_usage.total_tokens == 70
        assert tracker.total_usage.cost == 0.003

    def test_get_usage_by_provider(self):
        """Test getting usage grouped by provider."""
        tracker = TokenTracker()

        # Add usage records for different providers
        usage1 = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30, cost=0.001)
        usage2 = TokenUsage(prompt_tokens=15, completion_tokens=25, total_tokens=40, cost=0.002)
        usage3 = TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15, cost=0.0005)

        tracker.record_usage(token_usage=usage1, provider="openai", model="gpt-3.5-turbo", operation="op1")
        tracker.record_usage(token_usage=usage2, provider="gemini", model="gemini-1.5-flash", operation="op2")
        tracker.record_usage(token_usage=usage3, provider="openai", model="gpt-4", operation="op3")

        # Get usage by provider
        provider_usage = tracker.get_usage_by_provider()

        # Check that we have the right providers
        assert "openai" in provider_usage
        assert "gemini" in provider_usage

        # Check the usage for each provider
        assert provider_usage["openai"].prompt_tokens == 15
        assert provider_usage["openai"].completion_tokens == 30
        assert provider_usage["openai"].total_tokens == 45
        assert provider_usage["openai"].cost == 0.0015

        assert provider_usage["gemini"].prompt_tokens == 15
        assert provider_usage["gemini"].completion_tokens == 25
        assert provider_usage["gemini"].total_tokens == 40
        assert provider_usage["gemini"].cost == 0.002

    def test_get_usage_by_model(self):
        """Test getting usage grouped by model."""
        tracker = TokenTracker()

        # Add usage records for different models
        usage1 = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30, cost=0.001)
        usage2 = TokenUsage(prompt_tokens=15, completion_tokens=25, total_tokens=40, cost=0.002)
        usage3 = TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15, cost=0.0005)

        tracker.record_usage(token_usage=usage1, provider="openai", model="gpt-3.5-turbo", operation="op1")
        tracker.record_usage(token_usage=usage2, provider="gemini", model="gemini-1.5-flash", operation="op2")
        tracker.record_usage(token_usage=usage3, provider="openai", model="gpt-4", operation="op3")

        # Get usage by model
        model_usage = tracker.get_usage_by_model()

        # Check that we have the right models
        assert "gpt-3.5-turbo" in model_usage
        assert "gemini-1.5-flash" in model_usage
        assert "gpt-4" in model_usage

        # Check the usage for each model
        assert model_usage["gpt-3.5-turbo"].total_tokens == 30
        assert model_usage["gemini-1.5-flash"].total_tokens == 40
        assert model_usage["gpt-4"].total_tokens == 15

    def test_get_usage_by_topic(self):
        """Test getting usage grouped by topic."""
        tracker = TokenTracker()

        # Add usage records for different topics
        usage1 = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30, cost=0.001)
        usage2 = TokenUsage(prompt_tokens=15, completion_tokens=25, total_tokens=40, cost=0.002)
        usage3 = TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15, cost=0.0005)

        tracker.record_usage(token_usage=usage1, provider="openai", model="gpt-3.5-turbo", operation="op1", topic="topic1")
        tracker.record_usage(token_usage=usage2, provider="gemini", model="gemini-1.5-flash", operation="op2", topic="topic2")
        tracker.record_usage(token_usage=usage3, provider="openai", model="gpt-4", operation="op3", topic="topic1")

        # Get usage by topic
        topic_usage = tracker.get_usage_by_topic()

        # Check that we have the right topics
        assert "topic1" in topic_usage
        assert "topic2" in topic_usage

        # Check the usage for each topic
        assert topic_usage["topic1"].prompt_tokens == 15
        assert topic_usage["topic1"].completion_tokens == 30
        assert topic_usage["topic1"].total_tokens == 45
        assert topic_usage["topic1"].cost == 0.0015

        assert topic_usage["topic2"].prompt_tokens == 15
        assert topic_usage["topic2"].completion_tokens == 25
        assert topic_usage["topic2"].total_tokens == 40
        assert topic_usage["topic2"].cost == 0.002

    def test_get_usage_summary(self):
        """Test getting a summary of token usage."""
        tracker = TokenTracker()

        # Add some usage records
        usage1 = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30, cost=0.001)
        usage2 = TokenUsage(prompt_tokens=15, completion_tokens=25, total_tokens=40, cost=0.002)

        tracker.record_usage(token_usage=usage1, provider="openai", model="gpt-3.5-turbo", operation="op1")
        tracker.record_usage(token_usage=usage2, provider="gemini", model="gemini-1.5-flash", operation="op2")

        # Get the summary
        summary = tracker.get_usage_summary()

        # Check the summary values
        assert summary["total_tokens"] == 70
        assert summary["prompt_tokens"] == 25
        assert summary["completion_tokens"] == 45
        assert summary["total_cost"] == 0.003
        assert summary["operation_count"] == 2

        # Check the operations
        assert "op1" in summary["operations"]
        assert "op2" in summary["operations"]
        assert summary["operations"]["op1"] == 1
        assert summary["operations"]["op2"] == 1

        # Check the providers and models
        assert "openai" in summary["providers"]
        assert "gemini" in summary["providers"]
        assert "gpt-3.5-turbo" in summary["models"]
        assert "gemini-1.5-flash" in summary["models"]

    def test_generate_report(self):
        """Test generating a formatted report."""
        tracker = TokenTracker()

        # Add some usage records
        usage1 = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30, cost=0.001)
        usage2 = TokenUsage(prompt_tokens=15, completion_tokens=25, total_tokens=40, cost=0.002)

        test_time = datetime(2025, 1, 1, 12, 0, 0)
        # Patch datetime.now() to return a fixed time
        with mock.patch('llm_markdown_generator.token_tracker.datetime') as mock_datetime:
            mock_datetime.now.return_value = test_time
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            tracker.record_usage(token_usage=usage1, provider="openai", model="gpt-3.5-turbo", operation="op1")
            tracker.record_usage(token_usage=usage2, provider="gemini", model="gemini-1.5-flash", operation="op2")

        # Generate the report
        report = tracker.generate_report()

        # Check that the report contains expected information
        assert "Token Usage Report" in report
        assert "Total Operations: 2" in report
        assert "Total Tokens: 70" in report
        assert "Prompt Tokens: 25" in report
        assert "Completion Tokens: 45" in report
        assert "Estimated Total Cost: $0.003000" in report
        assert "openai: 30 tokens, Cost: $0.001000" in report
        assert "gemini: 40 tokens, Cost: $0.002000" in report
        assert "gpt-3.5-turbo: 30 tokens, Cost: $0.001000" in report
        assert "gemini-1.5-flash: 40 tokens, Cost: $0.002000" in report
        assert "op1: 1 calls" in report
        assert "op2: 1 calls" in report

    def test_generate_detailed_report(self):
        """Test generating a detailed report with record details."""
        tracker = TokenTracker()

        # Add some usage records
        usage1 = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30, cost=0.001)
        usage2 = TokenUsage(prompt_tokens=15, completion_tokens=25, total_tokens=40, cost=0.002)

        test_time = datetime(2025, 1, 1, 12, 0, 0)
        # Patch datetime.now() to return a fixed time
        with mock.patch('llm_markdown_generator.token_tracker.datetime') as mock_datetime:
            mock_datetime.now.return_value = test_time
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            tracker.record_usage(token_usage=usage1, provider="openai", model="gpt-3.5-turbo", operation="op1")
            tracker.record_usage(token_usage=usage2, provider="gemini", model="gemini-1.5-flash", operation="op2")

        # Generate the detailed report
        report = tracker.generate_report(detailed=True)

        # Check that the report contains the detailed records section
        assert "Detailed Records:" in report
        assert "2025-01-01 12:00:00" in report
        assert "op1 - openai/gpt-3.5-turbo: 30 tokens, Cost: $0.001000" in report
        assert "op2 - gemini/gemini-1.5-flash: 40 tokens, Cost: $0.002000" in report

    def test_log_to_file(self):
        """Test logging usage records to a file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "token_usage.log"
            tracker = TokenTracker(log_path=log_path)

            # Add some usage records
            usage1 = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30, cost=0.001)
            usage2 = TokenUsage(prompt_tokens=15, completion_tokens=25, total_tokens=40, cost=0.002)

            test_time = datetime(2025, 1, 1, 12, 0, 0)
            # Patch datetime.now() to return a fixed time
            with mock.patch('llm_markdown_generator.token_tracker.datetime') as mock_datetime:
                mock_datetime.now.return_value = test_time
                mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

                tracker.record_usage(token_usage=usage1, provider="openai", model="gpt-3.5-turbo", operation="op1")
                tracker.record_usage(token_usage=usage2, provider="gemini", model="gemini-1.5-flash", operation="op2")

            # Check that the log file was created and contains the records
            assert log_path.exists()

            with open(log_path, "r") as f:
                lines = f.readlines()

            assert len(lines) == 2

            # Parse the records from the file
            record1 = json.loads(lines[0])
            record2 = json.loads(lines[1])

            assert record1["prompt_tokens"] == 10
            assert record1["completion_tokens"] == 20
            assert record1["total_tokens"] == 30
            assert record1["cost"] == 0.001
            assert record1["provider"] == "openai"
            assert record1["model"] == "gpt-3.5-turbo"
            assert record1["operation"] == "op1"

            assert record2["prompt_tokens"] == 15
            assert record2["completion_tokens"] == 25
            assert record2["total_tokens"] == 40
            assert record2["cost"] == 0.002
            assert record2["provider"] == "gemini"
            assert record2["model"] == "gemini-1.5-flash"
            assert record2["operation"] == "op2"

    def test_load_records_from_file(self):
        """Test loading usage records from a file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "token_usage.log"

            # Create a log file with some records
            test_time = datetime(2025, 1, 1, 12, 0, 0).isoformat()

            with open(log_path, "w") as f:
                f.write(json.dumps({
                    "timestamp": test_time,
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                    "cost": 0.001,
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "operation": "op1",
                    "topic": "topic1",
                    "metadata": {}
                }) + "\n")

                f.write(json.dumps({
                    "timestamp": test_time,
                    "prompt_tokens": 15,
                    "completion_tokens": 25,
                    "total_tokens": 40,
                    "cost": 0.002,
                    "provider": "gemini",
                    "model": "gemini-1.5-flash",
                    "operation": "op2",
                    "topic": "topic2",
                    "metadata": {}
                }) + "\n")

            # Create a tracker with the log file
            tracker = TokenTracker(log_path=log_path)

            # Check that the records were loaded
            assert len(tracker.records) == 2

            # Check that the total usage was calculated correctly
            assert tracker.total_usage.prompt_tokens == 25
            assert tracker.total_usage.completion_tokens == 45
            assert tracker.total_usage.total_tokens == 70
            assert tracker.total_usage.cost == 0.003

            # Check some details of the loaded records
            assert tracker.records[0].provider == "openai"
            assert tracker.records[0].model == "gpt-3.5-turbo"
            assert tracker.records[0].topic == "topic1"

            assert tracker.records[1].provider == "gemini"
            assert tracker.records[1].model == "gemini-1.5-flash"
            assert tracker.records[1].topic == "topic2"

    def test_handle_malformed_log_file(self):
        """Test handling of malformed log files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "token_usage.log"

            # Create a log file with some valid and some invalid records
            with open(log_path, "w") as f:
                f.write(json.dumps({
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                    "provider": "openai"
                }) + "\n")

                # Invalid JSON
                f.write("This is not valid JSON\n")

                # Valid record
                f.write(json.dumps({
                    "prompt_tokens": 15,
                    "completion_tokens": 25,
                    "total_tokens": 40,
                    "provider": "gemini"
                }) + "\n")

            # Should not raise an exception
            tracker = TokenTracker(log_path=log_path)

            # Should have loaded the valid records
            assert len(tracker.records) == 2
            assert tracker.records[0].provider == "openai"
            assert tracker.records[1].provider == "gemini"

    def test_handle_nonexistent_log_file(self):
        """Test handling of a non-existent log file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "nonexistent.log"

            # Should not raise an exception
            tracker = TokenTracker(log_path=log_path)

            # Records should be empty
            assert len(tracker.records) == 0