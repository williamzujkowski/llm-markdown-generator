# 📁 Project File Tree
This document outlines the structure of the repository.
- **./**
    - .gitignore
    - .llmconfig -> llmconfig/ - *Symbolic link to llmconfig directory*
    - .pre-commit-config.yaml - *Pre-commit hooks configuration*
    - CODEOWNERS
    - CLAUDE.md - *Project guidance for Claude AI*
    - CONTRIBUTING.md
    - FILE_TREE.md - *This file - project structure overview*
    - LICENSE
    - pyproject.toml - *Project metadata and dependencies*
    - README.md
    - SECURITY.md
    - dependabot.yml
    - **config/** - *Configuration files*
        - config.yaml - *Main application configuration*
        - front_matter_schema.yaml - *Schema for markdown front matter*
    - **docs/** - *Documentation files*
        - .gitkeep
    - **llmconfig/** - *Contains LLM agent configurations and rules*
        - PROJECT_PLAN.md - *Project goals and roadmap*
        - USAGE_GUIDE.md
        - agent-rules.md
        - **context/** - *Context information for LLM agents*
            - .gitkeep
        - **examples/** - *Example scripts demonstrating framework usage*
            - README.md - *Overview of examples and usage*
            - error_handling_demo.py - *Demo of error handling features*
            - generate_with_pydantic.py - *Generate content with Pydantic configs*
            - test_plugin_system.py - *Demo of plugin system functionality*
            - generate_specialized_content.py - *Using different prompt templates*
            - **custom_plugins/** - *Custom plugin implementation examples*
                - __init__.py
                - example_plugins.py
        - **prompt-templates/** - *Templates for LLM interactions*
            - python_blog.j2 - *Template for Python blog posts*
            - javascript_blog.j2 - *Template for JavaScript blog posts*
            - data_science_blog.j2 - *Template for Data Science blog posts*
            - technical_tutorial.j2 - *Template for step-by-step technical guides*
            - product_review.j2 - *Template for detailed product reviews*
            - comparative_analysis.j2 - *Template for comparing technologies/frameworks*
            - research_summary.j2 - *Template for academic research summaries*
            - industry_trend_analysis.j2 - *Template for market and industry trend analysis*
            - security_advisory.j2 - *Template for security advisories on vulnerabilities*
            - daily_cve_report.j2 - *Template for daily reports on critical CVEs*
        - **system-prompts/** - *System-level instructions for different AI agents*
            - .gitkeep
    - **output/** - *Output directory for generated markdown files*
        - getting-started-with-python-type-hints.md - *Example generated blog post*
    - **scripts/** - *Utility scripts*
        - .gitkeep
    - **src/** - *Source code*
        - **llm_markdown_generator/** - *Main package directory*
            - __init__.py - *Package initialization*
            - cli.py - *Command line interface with Typer and Rich*
            - config.py - *Configuration handling*
            - config_pydantic.py - *Pydantic models for configuration validation*
            - error_handler.py - *Error handling with retries and backoff*
            - front_matter.py - *Front matter generation*
            - generator.py - *Main markdown generator*
            - llm_provider.py - *LLM providers (OpenAI and Google Gemini)*
            - prompt_engine.py - *Prompt template rendering*
            - token_tracker.py - *Token usage tracking and reporting*
            - **plugins/** - *Plugin system for content processing*
                - __init__.py - *Plugin loading and registration*
                - content_processor.py - *Content processing plugin interface*
                - front_matter_enhancer.py - *Front matter enhancement plugin interface*
    - **tests/** - *Test files*
        - __init__.py
        - **integration/** - *Integration tests*
            - __init__.py
            - test_api_keys.py - *Integration tests for API keys*
            - test_cli.py - *Tests for CLI functionality*
            - test_pipeline.py - *Tests for full generation pipeline*
            - test_plugin_integration.py - *Tests for plugin system integration*
            - test_token_tracking.py - *Tests for token tracking integration*
        - **unit/** - *Unit tests*
            - __init__.py
            - test_config.py - *Tests for config module*
            - test_config_pydantic.py - *Tests for Pydantic models*
            - test_error_handler.py - *Tests for error handling*
            - test_front_matter.py - *Tests for front_matter module*
            - test_generator.py - *Tests for generator module*
            - test_llm_provider.py - *Tests for llm_provider module*
            - test_plugins.py - *Tests for plugin system*
            - test_prompt_engine.py - *Tests for prompt_engine module*
            - test_token_tracker.py - *Tests for token usage tracking*