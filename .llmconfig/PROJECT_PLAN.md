# Project Plan & Goals: LLM Markdown Generator Framework

**Last Updated:** 2025-04-24

---

## 1. Overall Project Vision & Goal

To design and build a reusable, configurable, and extensible Python framework (tentatively named `llm-markdown-generator`) that leverages Large Language Models (LLMs) to generate markdown blog posts. The framework will specifically support customizable, 11ty-compatible YAML front matter and be suitable for generating content across a variety of topics, adhering strictly to the standards defined in `CLAUDE.md`. The goal is to automate and streamline the creation of structured blog content.

---

## 2. Current Phase Objectives (Initial Setup & Core Functionality)

* **Establish Project Structure:** Set up the repository according to `CLAUDE.md` guidelines (Section 6), including `src/`, `tests/`, `.llmconfig/`, `config/` directories, `pyproject.toml` (or `setup.py`), `venv`, `.gitignore`, `README.md`, `LICENSE`, `CONTRIBUTING.md`, and the critical `FILE_TREE.md`.
* **Implement Standard Tooling:** Configure `pre-commit` hooks with `black`, `isort`, `flake8`, `mypy` as defined in `CLAUDE.md` (Section 1 & 6.4).
* **Build Core Configuration System:** Implement loading of framework and topic settings from YAML files (`config/`). Include secure handling for LLM API keys (via environment variables/`.env`).
* **Develop Basic CLI:** Create a command-line interface (`argparse` or `typer`) to accept essential inputs (topic, config path, output path).
* **Implement Initial LLM Interface:** Define the `LLMProvider` interface/ABC and implement a client for *one* initial LLM provider (e.g., OpenAI or Anthropic).
* **Create Prompt Engine:** Implement logic to load and render prompts using Jinja2 templates stored in `.llmconfig/prompt-templates/`.
* **Develop Front Matter Generation:** Implement logic to create 11ty-compatible YAML front matter based on a configured schema.
* **Implement Basic Output Generation:** Combine front matter and LLM-generated content into a final `.md` file string and write it to disk.
* **Establish Unit Testing Foundation:** Write initial unit tests (using `pytest`) with mocking (`unittest.mock`) for core components (config loading, prompt rendering, LLM client mock, front matter generation). Aim for >85% coverage on initial components as per `CLAUDE.md` (Section 4.2.1).

---

## 3. Key Features / Framework Components Roadmap

*(Bold items are the focus for the Current Phase)*

* **Project Setup & Standards:**
    * **`CLAUDE.md` Compliant Repository Structure (`src`, `tests`, `.llmconfig`, `config`, `docs`, etc.)**
    * **`pyproject.toml` / `setup.py` for packaging and dev dependencies**
    * **Virtual Environment (`venv`) Setup**
    * **`pre-commit` hooks (black, isort, flake8, mypy)**
    * **`FILE_TREE.md` (maintained throughout)**
    * **Core Documentation (`README.md`, `LICENSE`, `CONTRIBUTING.md`)**
* **Command-Line Interface (CLI):**
    * **Basic CLI for generation (`argparse` or `typer`)**
    * Advanced CLI options (overrides, dry-run, etc.) (Future)
* **Configuration System:**
    * **Load main config from YAML (`PyYAML`)**
    * **Load topic-specific configs**
    * **Load front matter schemas**
    * **Secure API Key Handling (`python-dotenv`)**
    * Validation using Pydantic (Future)
* **Prompt Engine:**
    * **Load & Render Jinja2 Templates from `.llmconfig/prompt-templates/`**
    * **Context injection (topic, keywords, etc.)**
    * Support for prompt chaining or multiple prompt strategies per topic (Future)
* **LLM Interaction:**
    * **`LLMProvider` Interface/ABC**
    * **Initial LLM Client Implementation (e.g., OpenAI)**
    * Support for multiple LLM providers (Anthropic, Gemini, etc.) (Future)
    * **Basic API Error Handling**
    * Advanced Error Handling (Retries, Backoff) (Future)
    * Token usage tracking / Cost estimation (Future)
* **Response Processing:**
    * **Basic extraction of main content from LLM response**
    * Parsing structured data (JSON) from LLM for front matter population (Future)
* **Output Generation:**
    * **Front Matter Generator (11ty-compatible YAML)**
    * **Markdown Assembler (Front Matter + Content)**
    * **File Writer (with slugified filename generation)**
* **Testing (Pytest Framework):**
    * **Unit Tests for all core components**
    * **Use of Mocking (`unittest.mock`, `pytest-mock`)**
    * **High Code Coverage (`pytest-cov`, target >85% initial, >90% overall)**
    * Integration Tests (testing component interactions, mocking LLM API) (Future)
    * Property-Based Tests (e.g., for prompt generation logic) (Future)
    * Adherence to `CLAUDE.md` Testing Manifesto (Section 4)
* **Extensibility:**
    * Design for easy addition of new topics (via config)
    * Design for easy addition of new LLM Providers (Future)
    * Plugin system for custom logic (Future)
* **Documentation:**
    * **Google-style Docstrings for all public modules/functions/classes**
    * Comprehensive `README.md` (Usage, Config) (Future - build upon initial)
    * Detailed User Guide (`docs/`) (Future)
    * API Documentation Generation (e.g., Sphinx) (Future)

---

## 4. Target Audience / User Profile

Developers, content creators, and technical writers who need to generate structured markdown blog posts programmatically for various topics using LLMs, particularly for static site generators like 11ty that utilize YAML front matter. Users are expected to be comfortable with Python environments and configuring YAML files.

---

## 5. Core Principles & Constraints

* **Standard Compliance:** Strict adherence to all guidelines in `CLAUDE.md` (coding style, testing, documentation, security, repo structure).
* **Modularity & Extensibility:** Design components with clear interfaces (ABCs, Protocols) to allow replacement and extension.
* **Configuration Driven:** Core behaviors (prompts, front matter, LLM choice) should be controlled via external configuration files.
* **Testability:** Code must be designed for high testability (DI, mocking) enabling robust unit and integration tests.
* **Developer Experience:** The framework should be straightforward to set up, configure, and use for its core purpose. Extending it should be logical.
* **Security:** API credentials must be handled securely and not exposed in code or configuration files committed to version control. Input handling (if applicable later) must consider security.
* **Technology:** Python 3.x. Leverage standard libraries and well-maintained third-party packages.

---

## 6. Non-Goals (What This Framework Will NOT Provide Initially)

* **Graphical User Interface (GUI/Web UI):** Focus is on a CLI and library usage.
* **Integrated Content Editor:** The framework generates content; editing is done externally.
* **Image Generation or Handling:** Out of scope for the core framework.
* **Advanced SEO Tooling:** Beyond basic front matter fields like description/keywords if generated.
* **Language/Framework-Specific Project Setup:** Does not configure Node.js, specific Python web frameworks, etc., beyond its own Python package structure.
* **Deployment Pipelines:** Does not handle deploying the generated content to a website.

---

## 7. High-Level Architecture Notes

* The framework follows a pipeline architecture: Input (CLI) -> Configuration -> Prompt Generation -> LLM Interaction -> Response Processing -> Front Matter Generation -> Markdown Assembly -> File Output.
* Built entirely in Python, intended to be installable as a package (`pip install .`).
* Relies on external LLM APIs (via HTTP requests).
* Configuration managed via YAML files.
* LLM-specific instructions and prompts are centralized in the `.llmconfig/` directory.
* Code structure follows `CLAUDE.md` recommendations (`src/framework_name/`, `tests/`, etc.).
* Utilizes Dependency Injection patterns to facilitate testing and modularity.

---

Okay, here is Section 8 of the project plan, updated with links to the API documentation for Anthropic, Google Gemini, and OpenAI:

---

## 8. Links to Detailed Planning & Standards

* **Primary Standards Document:** `CLAUDE.md` (in repository root or `.llmconfig/`)
* **Project Task Tracking:** [Link to this repository's Issues tab]
* **Python Documentation:** [https://docs.python.org/3/](https://docs.python.org/3/)
* **Pytest Documentation:** [https://docs.pytest.org/](https://docs.pytest.org/)
* **11ty Front Matter Guide:** [https://www.11ty.dev/docs/data-frontmatter/](https://www.11ty.dev/docs/data-frontmatter/)
* **Anthropic (Claude) API Documentation:** [https://docs.anthropic.com/en/api/messages](https://docs.anthropic.com/en/api/messages) or [https://docs.anthropic.com/en/home](https://docs.anthropic.com/en/home)
* **Google (Gemini) API Documentation:** [https://ai.google.dev/gemini-api/docs](https://ai.google.dev/gemini-api/docs)
* **OpenAI (GPT) API Documentation:** [https://platform.openai.com/docs/api-reference](https://platform.openai.com/docs/api-reference) or [https://platform.openai.com/docs/api-reference/making-requests](https://platform.openai.com/docs/api-reference/making-requests)