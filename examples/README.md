# LLM Markdown Generator Examples

This directory contains example scripts demonstrating various features and use cases for the LLM Markdown Generator framework.

## Examples Overview

### 1. Custom Plugins

- **Directory**: `custom_plugins/`
- **Files**: `example_plugins.py`, `__init__.py`
- **Description**: Demonstrates how to create and use custom plugins for content processing and front matter enhancement.
- **Run Example**: 
  ```bash
  # Make sure you're in the main project directory
  python examples/test_plugin_system.py
  ```

### 2. Error Handling Demo

- **File**: `error_handling_demo.py`
- **Description**: Shows how the framework handles various error conditions like API errors, rate limits, etc.
- **Run Example**:
  ```bash
  python examples/error_handling_demo.py
  ```

### 3. Generate with Pydantic

- **File**: `generate_with_pydantic.py`
- **Description**: Demonstrates how to generate content with Pydantic-validated configurations.
- **Run Example**:
  ```bash
  python examples/generate_with_pydantic.py
  ```

### 4. Specialized Content Generation

- **File**: `generate_specialized_content.py`
- **Description**: Shows how to use the different specialized prompt templates for various content types.
- **Run Example**:
  ```bash
  # Generate a technical tutorial on setting up Docker with Python
  python examples/generate_specialized_content.py technical_tutorial "Setting up Docker with Python" --keywords "docker,python,containers,deployment" --audience "Python developers new to containerization"
  
  # Generate a product review for a programming tool
  python examples/generate_specialized_content.py product_review "VS Code for Python Development" --tone "detailed and balanced" --keywords "IDE,Python,productivity,extensions"
  
  # Generate a comparative analysis of web frameworks
  python examples/generate_specialized_content.py comparative_analysis "React vs Vue vs Angular" --audience "frontend developers" --keywords "frontend,frameworks,performance,ecosystem"
  
  # Generate a research summary on AI advancements
  python examples/generate_specialized_content.py research_summary "Recent Advances in Generative AI" --keywords "LLMs,diffusion models,multimodality,fine-tuning"
  
  # Generate an industry trend analysis
  python examples/generate_specialized_content.py industry_trend_analysis "Cloud Native Development Trends" --keywords "kubernetes,serverless,microservices,observability"
  ```

### 5. Security Report Generation

- **Files**: `generate_security_report.py`, `generate_cve_report.py`
- **Description**: Demonstrates how to generate daily CVE reports and security advisories for high-risk vulnerabilities.
- **Run Example**:
  ```bash
  # Generate a daily CVE report for today's critical vulnerabilities
  python examples/generate_security_report.py daily_cve_report today --verbose
  
  # Generate a daily CVE report for a specific date
  python examples/generate_security_report.py daily_cve_report 2025-04-15 --audience "security operations team" --keywords "RCE,zero-day,patch,Microsoft,Cisco"
  
  # Generate a security advisory for a specific CVE
  python examples/generate_security_report.py security_advisory "CVE-2024-21412" --keywords "VMware,authentication bypass,virtualization,critical infrastructure"
  
  # Generate a security advisory for a vulnerability trend with multiple CVEs
  python examples/generate_security_report.py security_advisory "Spring Framework Vulnerabilities" --cves "CVE-2024-12345,CVE-2024-12346,CVE-2024-12347" --audience "Java developers and system administrators"
  
  # Generate and save to a specific output directory
  python examples/generate_security_report.py daily_cve_report today --output-dir "security/reports"
  
  # Using the CLI command to generate a CVE report (simpler approach)
  python examples/generate_cve_report.py CVE-2023-12345
  
  # Using the CLI command with additional options
  python -m llm_markdown_generator.cli generate-cve-report CVE-2023-12345 --provider openai --model gpt-4o --output-dir "security/advisories"
  ```

## Running the Examples

Most examples can be run directly from the main project directory. Make sure you've set up your environment variables (API keys) before running examples that make API calls.

```bash
# Make sure you're in the project's root directory
cd /path/to/llm-markdown-generator

# Set up your environment variables
export OPENAI_API_KEY=your_openai_key
# OR
export GEMINI_API_KEY=your_gemini_key

# Run an example
python examples/generate_specialized_content.py technical_tutorial "Python Type Hints" --verbose
```

## Additional Notes

- These examples are designed to showcase specific features of the framework
- The examples assume you have properly installed the package or are running them from the project directory
- Check each example's source code for additional command-line options and configurations