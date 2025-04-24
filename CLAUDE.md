# CLAUDE.md (Consolidated Project Guidance for Claude)

This file provides comprehensive guidance for Claude AI when working with code in this repository. It consolidates project-specific setup, coding standards, testing methodologies, and overall project principles.

**Purpose:** Use this file as a primary reference for understanding project conventions, requirements, and best practices when generating or modifying code.

**Important Notes:** 
1. While this file provides general guidance, all prompt templates and LLM-specific configurations should be stored in the `.llmconfig/` directory to ensure consistent AI agent behavior across the project.
2. The `FILE_TREE.md` document MUST be kept updated whenever the repository structure changes to ensure AI agents can effectively navigate the codebase.

**Content Overview:**
1.  **Quick Start:** Build, setup, lint, and test commands.
2.  **Project Code Style Summary:** Key style guidelines for this specific project.
3.  **Detailed Coding Standards:** Comprehensive rules covering style, documentation, architecture, security, performance, and more.
4.  **Detailed Testing Manifesto:** In-depth testing principles and quality assurance standards.
5.  **Overall Project Standards Framework:** High-level view integrating development lifecycle, AI ethics, technical quality, and operations.
6.  **Repository Structure for LLM Code Agents:** Guidelines for organizing repositories to optimize for LLM-based development.
7.  **Master Prompts:** Pre-defined prompts for guiding LLM code generation based on these standards.

---

## 1. Quick Start: Build, Setup, Lint & Test Commands

### Build & Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Lint & Test
```bash
# Format code
black .
isort .

# Lint code
flake8
mypy .

# Run all tests
pytest

# Run a single test
pytest tests/path/to/test_file.py::test_function_name

# Run tests with coverage
pytest --cov=src
```

---

## 2. Project Code Style Summary

* **Base Standard:** PEP 8.
* **Line Length:** Max 88 characters (Black default).
* **Naming:**
    * Variables, functions, methods: `snake_case` (e.g., `user_count`, `calculate_total()`).
    * Classes: `PascalCase` (e.g., `DependencyParser`, `RiskProfiler`).
    * Constants: `UPPER_SNAKE_CASE`.
* **Type Annotations:** Required for all function and method signatures.
* **Docstrings:** Use Google-style docstrings.
* **Function Size:** Aim for focused functions, ideally under 50 lines.
* **Exceptions:** Prefer custom exception classes for specific error conditions.
* **Imports:** Organize imports: standard library, third-party, local application (handled by `isort`).

---

## 3. Detailed Coding Standards

These standards provide comprehensive guidelines beyond the project summary.

### 3.1. Code Style and Formatting
1.  Follow established style guides for your language (e.g., Python: PEP 8).
2.  Enforce consistent formatting (indentation, line length (88 chars), statement termination, brackets, whitespace).
3.  Use meaningful naming conventions (PascalCase classes, snake\_case functions/variables, UPPER\_SNAKE\_CASE constants, underscore prefix for private).
4.  Structure code consistently (organize imports, group related functions/methods, consistent file organization, limit file/function size).
5.  Automate style enforcement (linters like flake8, formatters like black, pre-commit hooks, CI/CD integration).

### 3.2. Documentation Standards
1.  Include documentation for all public interfaces (purpose, parameters, returns, exceptions, examples).
2.  Add contextual documentation (module/file/class level, complex algorithms, rationale for decisions).
3.  Follow documentation format standards (consistent docstring format like Google-style, type hints, side effects, thread safety, performance notes).
4.  Maintain system-level documentation (architecture diagrams, flows, data models, APIs, deployment).
5.  Establish documentation review process (during code reviews, test accuracy, update with code, track coverage).

### 3.3. Architecture and Design Patterns
1.  Establish clear architectural boundaries (layers, separation of concerns, dependency inversion, module responsibilities, document decisions).
2.  Apply appropriate design patterns (Creational, Structural, Behavioral, Concurrency, Domain-specific).
3.  Follow SOLID principles (Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion).
4.  Design for extensibility (interfaces, plugin systems, extension points, avoid tight coupling, feature toggles).
5.  Establish system boundaries (clear APIs, encapsulation, DDD where applicable, document constraints/assumptions, catalog technical debt).

### 3.4. Security Best Practices
1.  Apply input validation (validate at entry, sanitize, parameterized queries, safe parsers, type safety).
2.  Implement proper authentication (OAuth 2.0/OIDC, secure credential storage, MFA, session management, password policies).
3.  Apply appropriate authorization (RBAC/ABAC, check at each layer, least privilege, capability-based security, audit decisions).
4.  Protect sensitive data (encrypt at rest/transit, key management, minimize collection, retention policies, secure logging).
5.  Apply secure coding practices (secure defaults, proper error handling, memory safety, secure dependencies, security headers).

### 3.5. Performance Optimization
1.  Establish performance targets (response time, throughput, resource usage, latency, scaling).
2.  Apply algorithmic efficiency (appropriate data structures/algorithms, complexity analysis, avoid O(n^2+), memoization).
3.  Implement resource optimization (connection pooling, caching, memory management, I/O optimization, async processing).
4.  Apply database optimization (efficient schemas, indexes, optimized queries, query caching, DB-specific features).
5.  Implement proper benchmarking (automated tests, baselines, track metrics, profiling, continuous monitoring).

### 3.6. Error Handling
1.  Define error handling strategy (recoverable vs non-recoverable, exception hierarchies, retry policies, document behavior, reporting).
2.  Implement defensive programming (check preconditions, validate arguments, handle edge cases, design for failure, assertions).
3.  Create informative error messages (context, troubleshooting guidance, consistent format, error codes, localization).
4.  Apply proper exception handling (catch specific exceptions, avoid empty catch, maintain context, cleanup resources, log info).
5.  Implement structured logging (errors with stack traces, correlation IDs, severity levels, context-aware, avoid sensitive info).

### 3.7. Resource Management
1.  Apply proper resource lifecycle management (acquire late, release early, pooling, timeouts, circuit breakers).
2.  Handle external dependencies gracefully (fallbacks, progressive degradation, bulkheading, health checks, partial availability).
3.  Implement proper concurrency control (locking, synchronization, prevent deadlocks/races, pooling, non-blocking algorithms).
4.  Manage memory efficiently (cleanup, avoid leaks, resource limits, weak references, profile usage).
5.  Optimize file and network operations (buffer I/O, non-blocking I/O, connection pooling, batching, streaming).

### 3.8. Dependency Management
1.  Define dependency selection criteria (license, security, maintenance, community, compatibility).
2.  Implement version pinning (lock direct dependencies, specify ranges, document rationale, regular updates, automate updates).
3.  Apply dependency isolation (virtual environments, containerization, dependency injection, manage transitives, minimize footprint).
4.  Implement vulnerability scanning (CI/CD integration, security advisories, auto-updates, inventory, mitigation strategies).
5.  Create dependency documentation (purpose, mapping to features, alternatives analysis, upgrade paths, custom patches).

### 3.9. Version Control Practices (Git)
1.  Define branch management strategy (trunk-based/GitFlow, naming conventions, lifetimes, merge requirements, protection rules).
2.  Create commit standards (descriptive messages, conventional commits format, issue refs, atomic commits, sign commits).
3.  Implement code review workflows (pull/merge requests, review criteria, approvers, automated checks, review responsibilities).
4.  Apply versioning standards (semantic versioning, document breaking changes, changelogs, tag releases, archive artifacts).
5.  Establish repository hygiene (.gitignore, artifact storage, repo structure docs, hooks, repo documentation).

### 3.10. Code Review Standards
1.  Define review scope (correctness, standards adherence, security, performance, documentation).
2.  Establish review process (reviewers, size limits, timeframes, checklists, roles).
3.  Apply technical review criteria (algorithm correctness, error handling, test coverage, maintainability, compatibility).
4.  Implement review automation (style checks, static analysis, build/test verification, doc coverage, quality metrics).
5.  Foster constructive review culture (focus on code, actionable feedback, explain reasoning, ask questions, acknowledge good work).

### 3.11. Accessibility Standards (If Applicable)
1.  Apply semantic structure (HTML elements, headings, ARIA, labels, keyboard navigation).
2.  Implement responsive design (screen sizes, touch targets, flexible layouts, responsive images, multi-device testing).
3.  Apply color and contrast standards (WCAG contrast, don't rely on color alone, focus indicators, high contrast modes, simulators).
4.  Implement assistive technology support (alt text, transcripts, captions, screen reader support, testing).
5.  Apply accessibility testing (automated checkers, manual keyboard/screen reader testing, include in QA, document features).

### 3.12. Internationalization & Localization (If Applicable)
1.  Apply proper text externalization (extract strings, avoid concatenation, templates, pluralization, gender).
2.  Implement locale awareness (date/time/number formats, currencies, text directionality, sorting).
3.  Design for text expansion (flexible UI, avoid fixed width, test longer languages, dynamic layout, character sets).
4.  Implement resource management (organize by locale, fallbacks, efficient loading, locale switching, translation workflow).
5.  Apply localization testing (pseudo-localization, verify locales, bidirectional text, cultural appropriateness, native speaker testing).

### 3.13. Concurrency and Parallelism
1.  Define concurrency models (threading, async/await, actors, event-driven, data access patterns).
2.  Implement thread safety (document guarantees, use thread-safe collections, synchronization, immutability, atomics).
3.  Apply parallelism patterns (task/data/pipeline parallelism, work distribution, scale-out).
4.  Manage shared resources (document contention, locks, lock-free algorithms, resource limits, backpressure).
5.  Test concurrent code (race conditions, load testing, simulate slow resources, fuzzing, document assumptions).

### 3.14. API Design
1.  Apply API design principles (consistency, intuitiveness, least surprise, evolution, document decisions).
2.  Implement proper versioning (semantic, backward compatibility, document breaking changes, migration paths, graceful deprecation).
3.  Define interface contracts (behavior, constraints, errors, side effects, performance).
4.  Apply REST/GraphQL best practices (HTTP methods, naming, status codes, efficient queries, pagination/filtering).
5.  Implement API security (authentication, authorization, rate limiting, input validation, document requirements).

### 3.15. Refactoring Guidelines
1.  Define refactoring triggers (code smells, complexity, performance bottlenecks, tech debt, maintainability metrics).
2.  Establish refactoring processes (document behavior, create tests, small changes, review, verify preservation).
3.  Implement refactoring techniques (extract method/class, consolidate conditionals, composition over inheritance, patterns, simplify).
4.  Apply refactoring tools (IDE features, automated tools, static analysis, quality metrics, history).
5.  Document refactoring outcomes (measure improvements, update docs, lessons learned, debt reduction, new patterns).

### 3.16. Sustainability and Green Coding
1.  Optimize resource efficiency (CPU, memory, I/O, lazy loading, efficient algorithms).
2.  Apply energy-aware design (batch background ops, efficient polling, push vs pull, mobile battery, power profiles).
3.  Implement efficient data practices (minimize transfers, compression, caching, optimize media, efficient formats).
4.  Design for hardware efficiency (support older hardware, progressive enhancement, minimize intensive animations, hardware acceleration).
5.  Measure environmental impact (energy tracking, carbon footprint, green hosting, document improvements, include efficiency in metrics).

---

## 4. Detailed Testing Manifesto

Follow these principles and standards for comprehensive testing.

### 4.1 Core Testing Principles

#### 4.1.1. Hypothesis Tests for Behavior Validation
* Identify the core hypothesis for each function.
* Write tests defining clear expectations ("Given X, should do Y").
* Test positive, negative, and boundary cases.
* Verify error handling.
* Use descriptive test names.
```python
# Example
def test_user_authentication_valid_credentials():
    """HYPOTHESIS: Given valid credentials, authentication should succeed."""
    # Arrange
    valid_username = "test_user"
    valid_password = "correct_password"
    # Act
    result = authenticate_user(valid_username, valid_password)
    # Assert
    assert result.success is True
    assert result.error_message is None
```

#### 4.1.2. Regression Tests for Known Fail States
* For each bug fix, create a test recreating failure conditions.
* Document the original issue and fix verification.
* Include issue/ticket references.
* Maintain a dedicated regression suite.
```python
# Example
def test_calculation_with_zero_division_protection():
    """REGRESSION: Bug #1234 - Division by zero crash.
    Ensures safe_divide returns None instead of raising ZeroDivisionError."""
    # Arrange
    input_value = 10
    divisor = 0
    expected_result = None
    # Act
    result = safe_divide(input_value, divisor)
    # Assert
    assert result == expected_result
```

#### 4.1.3. Benchmark Tests with SLA Enforcement
* Define clear performance metrics and SLAs (latency, throughput, resources, errors).
* Create tests measuring against defined thresholds in controlled environments.
* Include average and percentile measurements (p95, p99).
* Alert on violations.
```python
# Example (Conceptual)
def test_api_response_time_sla():
    """BENCHMARK: API p95 < 200ms, p99 < 500ms, error rate < 0.1%."""
    # Arrange: Setup load generation
    # Act: Run N requests, collect timings and errors
    # Assert: Calculate p95, p99, error_rate and compare against SLA
    pass # Implementation depends on benchmark tool (e.g., pytest-benchmark, locust)
```

#### 4.1.4. Grammatical Evolution (GE) for Fuzzing + Edge Discovery
* Define a grammar (e.g., BNF) for valid system inputs.
* Implement an evolutionary algorithm to generate grammatically correct but potentially unexpected test cases.
* Use fitness functions prioritizing edge cases and coverage.
* Log failures and add discovered edge cases to regression tests.
```python
# Example (Conceptual)
def test_with_grammatical_evolution():
    """FUZZING: Use GE to discover edge cases in the input parser."""
    # Arrange: Define grammar, configure GE fuzzer
    # Act: Run fuzzer against target function (e.g., api_request_handler)
    # Assert: Check for critical failures, add findings to regression suite.
    pass # Requires a GE framework implementation
```

#### 4.1.5. Structured Logs for Agent Feedback
* Design structured logging (e.g., JSON) capturing inputs, outputs, decisions, confidence scores, metrics, deviations.
* Use consistent formats and correlation IDs.
* Implement levels for filtering.
* Create log analyzers.
```python
# Example Test
def test_agent_logging_completeness():
    """AGENT FEEDBACK: Verify agent produces comprehensive structured logs."""
    # Arrange: Define expected log fields, setup log capture
    # Act: Run agent process
    # Assert: Check logs exist, required fields are present, sequence complete, decisions logged.
    pass # Requires agent instrumentation and log capture mechanism
```

### 4.2 Quality Assurance Standards

#### 4.2.1. Code Coverage Requirements
* Thresholds: 85%+ overall line coverage, 95%+ critical components, 100% utilities.
* Track trends, prevent regression.
* Cover all paths, branches, error handling.
* Integrate reports in CI/CD, block merges below thresholds.
```python
# Example Config (e.g., pyproject.toml for pytest-cov)
# [tool.coverage.run]
# branch = true
# source = ["src"]
# [tool.coverage.report]
# fail_under = 85
# show_missing = true
# # Configure path-specific thresholds if tool supports it
```

#### 4.2.2. Static Analysis Rules
* Configure linters (flake8, pylint) and analyzers (mypy, bandit) with strict rules.
* Enforce style, find bugs, identify security/performance issues.
* Create project-specific custom rules.
* Integrate into pre-commit hooks and CI/CD.
* Maintain a "zero warnings" policy (treat warnings as errors).
```python
# Example pre-commit hook config (`.pre-commit-config.yaml`)
# - repo: [https://github.com/pycqa/flake8](https://github.com/pycqa/flake8)
#   rev: ...
#   hooks:
#   - id: flake8
#     additional_dependencies: [flake8-bugbear, flake8-comprehensions, ...]
# - repo: [https://github.com/pre-commit/mirrors-mypy](https://github.com/pre-commit/mirrors-mypy)
#   rev: ...
#   hooks:
#   - id: mypy
```

#### 4.2.3. Contract Testing Framework
* Define explicit contracts for service interfaces (request/response formats, errors, performance, versions).
* Implement consumer-driven contract tests (e.g., using Pact).
* Maintain a contract registry.
* Automate verification in CI/CD, alert on violations.
```python
# Example (Conceptual using Pact)
# Consumer Side: Define expectations, generate pact file.
# Provider Side: Verify pact file against provider implementation.
# CI: Publish pacts, trigger provider verification.
```

#### 4.2.4. Mutation Testing Guidelines
* Apply systematic code mutations (modify operators, boundaries, logic) to test test effectiveness.
* Establish mutation score thresholds (e.g., 80%+ overall, 90%+ critical).
* Integrate periodically into CI/CD or on significant changes.
* Review and address surviving mutants.
* Focus on high-value targets (business logic, security, error handling).
```python
# Example (Using mutmut)
# Run: mutmut run
# Check results: mutmut results
# Aim for high killed mutant score. Add tests for surviving mutants.
```

#### 4.2.5. Property-Based Testing Framework
* Define invariant properties the code must satisfy (reversibility, mathematical laws, business rules).
* Use libraries like Hypothesis to generate diverse test inputs automatically.
* Define explicit property assertions focusing on "what" not "how".
* Incorporate failure analysis and add specific regressions for found issues.
```python
# Example (Using Hypothesis)
from hypothesis import given, strategies as st

@given(st.lists(st.integers()))
def test_sort_idempotence(values):
    """PROPERTY: Sorting an already sorted list produces the same result."""
    once = sorted(values)
    twice = sorted(once)
    assert once == twice
```

### 4.3 Security and Resilience

#### 4.3.1. Security Testing Guidelines
* Apply multi-level security testing (SAST, DAST, IAST, dependency scanning).
* Test against standards (OWASP Top 10, CWE/SANS Top 25).
* Implement specific tests for common vulnerabilities (injection, auth bypass, data exposure, XXE, SSRF, misconfigs).
* Incorporate into CI/CD, block on critical findings, track security debt.
```python
# Example
def test_sql_injection_prevention():
    """Verify protection against SQL injection attacks using known vectors."""
    # Arrange: Define attack vectors
    # Act/Assert: Test vulnerable endpoints with attack vectors, ensure no data leak/unintended action.
    pass # Use security testing tools/frameworks where possible
```

#### 4.3.2. Resilience Testing Framework (Chaos Engineering)
* Design chaos experiments (dependency failures, network issues, resource exhaustion).
* Define steady-state hypotheses and acceptable degradation/recovery.
* Run controlled experiments in production-like environments with monitoring and abort conditions.
* Incorporate findings into architecture and add regression tests.
```python
# Example (Conceptual)
def test_resilience_to_database_failure():
    """Verify system resilience (e.g., fallback to cache) when DB is unavailable."""
    # Arrange: Define steady state check
    # Act: Simulate DB failure (using chaos tooling or mocks)
    # Assert: Verify system behavior (e.g., reads work from cache, writes fail gracefully)
    # Act: Restore DB connection
    # Assert: Verify system recovers to steady state within RTO.
```

### 4.4 Documentation and Integration

#### 4.4.1. Documentation Testing
* Test all code examples in documentation (extract and execute).
* Validate API documentation completeness against actual implementation.
* Test user journeys (tutorials, setup instructions).
* Automate doc tests in CI/CD, flag documentation drift.
```python
# Example (Using doctest or custom script)
# Extract code blocks from Markdown/RestructuredText.
# Execute each block and check output against expected results if specified.
```

#### 4.4.2. Integration Testing Patterns
* Define integration boundaries explicitly.
* Use appropriate test doubles (mocks, stubs, spies) or real instances strategically.
* Implement end-to-end test scenarios for key user journeys/workflows.
* Manage test environments effectively (containerization, state reset, parallelization).
```python
# Example
def test_user_registration_end_to_end():
    """Verify the complete user registration process across all components."""
    # Arrange: Setup necessary services (API, DB, Email service mock/test mode)
    # Act: Perform registration steps via API calls
    # Assert: Verify intermediate steps (e.g., email sent) and final state (e.g., user in DB, login works).
```

#### 4.4.3. Testability Guidelines
* Design for testability (dependency injection, separation of concerns, avoid global state, pure functions).
* Create testability interfaces (time/filesystem/network abstractions, randomness control).
* Implement testing hooks (instrumentation, state inspection).
* Include testability in code reviews, refactor hard-to-test code.
```python
# Example: Use Dependency Injection
class TestableService:
    # Inject dependencies instead of creating them internally
    def __init__(self, db_client, email_notifier):
        self.db_client = db_client
        self.email_notifier = email_notifier
    # ... methods using injected dependencies ...

# In tests, inject mock/stub implementations
mock_db = MockDbClient()
mock_email = MockEmailNotifier()
service = TestableService(mock_db, mock_email)
# Test service logic...
```

---

## 5. Overall Project Standards Framework

This outlines the high-level framework guiding the project:

1.  **Development Lifecycle:** Tailored methodologies for LLM projects, knowledge sharing for AI specifics, cross-functional team collaboration, documentation for models/data/systems.
2.  **AI Ethics and Governance:** Responsible AI guidelines, bias detection/mitigation, privacy standards, transparent decision documentation, human oversight implementation.
3.  **Technical Quality:** Testing standards (as detailed above), coding standards (as detailed above), data governance, AI system deployment standards, model performance monitoring/observability.
4.  **Operational Excellence:** Incident management for AI issues, MLOps practices, AI regulation compliance, AI interface UX standards, sustainability considerations.

*(This section provides context; detailed implementation is covered in Coding and Testing standards sections.)*

---

## 6. Repository Structure for LLM Code Agents

Designing a repository optimized for LLM-driven code agents involves establishing a structured, maintainable, and secure environment that facilitates high-quality code generation and adherence to established standards.

### 6.1. Root-Level Files and Directories

- **`README.md`**: Provides an overview of the project, including its purpose, setup instructions, and usage examples
- **`CONTRIBUTING.md`**: Outlines guidelines for contributing to the project, including code standards, branching strategies, and pull request procedures
- **`CODE_OF_CONDUCT.md`**: Defines expected behavior for contributors to foster an inclusive and respectful community
- **`LICENSE`**: Specifies the project's licensing information
- **`FILE_TREE.md`**: **CRITICAL** - Maintains an up-to-date file structure overview of the project. This file must be kept current whenever the repository structure changes to ensure AI agents can properly navigate the codebase.
- **`docs/`**: Contains detailed documentation, such as architecture overviews, API references, and style guides
- **`src/`**: Houses the main source code, organized by feature or module
- **`tests/`**: Includes unit, integration, and end-to-end tests, mirroring the structure of the `src/` directory
- **`scripts/`**: Contains utility scripts for tasks like setup, deployment, and maintenance
- **`config/`**: Stores configuration files for different environments (e.g., development, testing, production)
- **`.github/`**: Includes GitHub-specific files, such as issue templates and workflows
- **`.llmconfig/`**: Dedicated directory for LLM agent configurations and rules

### 6.2. LLM Agent Configuration (`.llmconfig/`)

**IMPORTANT:** All LLM-specific configurations, prompts, and context files MUST be stored in the `.llmconfig/` directory rather than in other locations such as project root. This ensures consistent AI agent behavior and makes prompt engineering maintainable.

The `.llmconfig/` directory should contain:

- **`CLAUDE.md`**: Primary configuration file for Claude AI, containing general instructions
- **`agent-rules.md`**: Defines coding standards and best practices for the LLM agent to follow
- **`prompt-templates/`**: Contains reusable prompt templates to guide the LLM's behavior
- **`context/`**: Includes files that provide context to the LLM, such as sample inputs/outputs and domain-specific information
- **`examples/`**: Contains exemplary interactions or code snippets for few-shot learning
- **`system-prompts/`**: Holds system-level instructions for different AI agents

This structured approach ensures that all prompt engineering work is centralized, versioned, and properly maintained alongside the codebase.

### 6.3. Maintaining FILE_TREE.md

**CRITICAL:** Whenever the repository structure changes (adding, removing, or moving files and directories), the `FILE_TREE.md` document must be updated accordingly. This file serves as a map of the codebase for LLM agents and is essential for their ability to:

1. Efficiently navigate the project structure
2. Understand relationships between components
3. Identify relevant files when implementing new features or fixing bugs
4. Maintain a mental model of the overall architecture

The `FILE_TREE.md` should:
- Provide a hierarchical representation of the repository's file structure
- Include brief descriptions for key directories and files
- Highlight important files that define core functionality
- Be automatically updated by scripts when possible, or manually updated after structural changes
- Be referenced in commit messages when updated ("Update FILE_TREE.md to reflect new module structure")

### 6.4. Configuration and Tooling

#### 6.4.1. Linters and Formatters
- **JavaScript/TypeScript**: Use ESLint and Prettier with a shared configuration to enforce code style and formatting
- **Python**: Employ Flake8 and Black for linting and formatting, respectively
- **Go**: Utilize `go fmt` and `golint` to maintain code consistency

#### 6.4.2. Type Checking
- **TypeScript**: Enable strict mode in `tsconfig.json` to enforce rigorous type checking
- **Python**: Incorporate type hints and use tools like `mypy` for static type analysis

#### 6.4.3. Testing Frameworks
- **JavaScript/TypeScript**: Implement Jest for unit and integration testing
- **Python**: Use Pytest to write and manage tests
- **Go**: Leverage the built-in `testing` package for writing tests

#### 6.4.4. Continuous Integration/Continuous Deployment (CI/CD)
- **GitHub Actions**: Set up workflows to automate testing, linting, and deployment processes
- **Pre-commit Hooks**: Use tools like `pre-commit` to run linters and tests before commits are made

### 6.5. Coding Standards and Best Practices

#### 6.5.1. Naming Conventions
- **Variables and Functions**: Use `camelCase` for naming
- **Classes and Types**: Adopt `PascalCase`
- **Constants**: Use `UPPER_SNAKE_CASE`

#### 6.5.2. Code Structure
- **Modularity**: Design code in small, reusable modules or functions
- **Single Responsibility**: Ensure each module or function has a single, well-defined purpose
- **Documentation**: Include docstrings or comments to explain complex logic and usage

### 6.6. Security and Compliance

- **Input Sanitization**: Ensure all user inputs are properly sanitized to prevent injection attacks
- **Dependency Management**: Regularly update dependencies and monitor for known vulnerabilities using tools like `npm audit` or `pip-audit`
- **Secrets Management**: Store secrets and sensitive information securely, avoiding hardcoding them in the codebase
- **Access Controls**: Implement proper authentication and authorization mechanisms to protect resources

### 6.7. Testing and Quality Assurance

- **Test Coverage**: Aim for high test coverage, focusing on critical and complex parts of the codebase
- **Automated Testing**: Integrate automated tests into the CI/CD pipeline to catch regressions early
- **Code Reviews**: Conduct thorough code reviews to maintain code quality and share knowledge among team members

### 6.8. Documentation and Communication

- **API Documentation**: Use tools like Swagger or JSDoc to generate and maintain API documentation
- **Changelog**: Maintain a `CHANGELOG.md` to document significant changes and updates
- **Architecture Diagrams**: Include diagrams to visualize system architecture and data flow
- **FILE_TREE.md**: Keep this file updated with each structural change to the repository (this is a critical requirement for LLM agents)

---

## 7. Master Prompts for LLM Interaction

Use these prompts as starting points when requesting code generation or modification:

### Master Prompt for Coding Standards Implementation
```
Generate code for [specific feature/module description].

**IMPORTANT CONTEXT:** Before starting, review the `PROJECT_PLAN.md` file located in the repository root. Ensure the code you generate aligns with the overall project vision, current phase objectives, and core principles outlined there. Keep the target audience and non-goals in mind when making implementation choices.

Follow these comprehensive coding standards:

1.  **Style and Structure:**
    * Follow Python PEP 8 style guide conventions with Black formatting (88 char lines).
    * Use meaningful, consistent naming (snake\_case for functions/vars, PascalCase for classes).
    * Document all public interfaces thoroughly using Google-style docstrings with type hints.
    * Limit function/method size ideally below 50 lines.
    * Apply consistent error handling using custom exceptions where appropriate.

2.  **Architecture and Design:**
    * Implement SOLID principles.
    * Apply appropriate design patterns where beneficial.
    * Define clear component boundaries and use dependency injection.
    * Design testable components.
    * Document significant architectural decisions.

3.  **Security and Performance:**
    * Validate all inputs thoroughly.
    * Apply proper authentication/authorization if relevant.
    * Optimize critical algorithms and resource usage.
    * Manage resources efficiently (e.g., use context managers).
    * Implement appropriate caching if needed.

4.  **Quality and Maintenance:**
    * Create comprehensive tests alongside the code (refer to Testing Manifesto in CLAUDE.md).
    * Document complex logic clearly.
    * Apply internationalization/accessibility standards if required by the feature.
    * Design for extensibility.

The code should be robust, efficient, secure, maintainable, directly contribute to the objectives in `PROJECT_PLAN.md`, and demonstrate high craftsmanship adhering to all project standards outlined in `CLAUDE.md`.

Note: This is a template example. For project use, store actual prompt templates in the `.llmconfig/prompt-templates/` directory.
```

### Master Prompt for Test Suite Generation
```
Generate a comprehensive test suite for the provided code ([link/path to code file(s)]).

**IMPORTANT CONTEXT:** Before starting, review the `PROJECT_PLAN.md` file located in the repository root. Ensure the tests you generate verify that the code behaves according to the overall project vision, current phase objectives, and core principles outlined there. Test scenarios should reflect the target audience's usage patterns and respect defined non-goals.

Follow the project's Testing Manifesto (detailed in CLAUDE.md):

1.  **Core Testing Principles:**
    * Hypothesis tests validating core behaviors and expectations relevant to the project goals.
    * Regression tests covering known edge cases or previously fixed bugs.
    * Benchmark tests for performance-critical sections (if applicable, aligning with principles in `PROJECT_PLAN.md`).
    * Consider inputs for fuzzing/edge case discovery (if applicable).
    * Ensure code generates structured logs if it's an agent/process.

2.  **Quality Assurance:**
    * Aim for high code coverage (85%+ general, 95%+ critical).
    * Ensure code passes static analysis (flake8, mypy).
    * Implement contract tests if interacting with defined interfaces.
    * Consider property-based tests (using Hypothesis) for functions with clear invariants.

3.  **Security and Resilience:**
    * Include security tests relevant to project constraints and potential risks.
    * Add resilience tests if the code handles external dependencies or failure modes mentioned as constraints/principles.

4.  **Documentation and Integration:**
    * Create integration tests for interactions between components/modules if applicable to the current objectives.
    * Ensure tests are clear and serve as documentation.

For each test:
* Use descriptive names (`test_feature_scenario_expected_outcome`).
* Include clear docstrings explaining the test's purpose and its relevance to project goals if not obvious.
* Arrange preconditions, Act by calling the code, Assert expected outcomes.
* Use fixtures (e.g., pytest fixtures) for setup/teardown where appropriate.
* Categorize tests appropriately (unit, integration, etc. - potentially via markers).

The test suite should be maintainable, provide fast feedback, and verify the code's correctness, robustness, and alignment with the requirements and goals specified in both `CLAUDE.md` and `PROJECT_PLAN.md`.

Note: This is a template example. For project use, store actual prompt templates in the `.llmconfig/prompt-templates/` directory.
```

---
**End of CLAUDE.md**