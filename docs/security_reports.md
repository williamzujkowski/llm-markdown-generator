# Security Reports and CVE Advisories

This document explains how to use the LLM Markdown Generator to create comprehensive security reports and CVE advisories.

## Overview

The LLM Markdown Generator provides specialized capabilities for generating security content using the `generate-cve-report` command. This feature allows you to create detailed security advisories for specific CVE IDs, with comprehensive information about vulnerabilities, their impact, and mitigation steps.

## Usage

### Basic Command

Generate a security advisory for a specific CVE:

```bash
llm-markdown generate-cve-report CVE-2023-12345
```

This will create a markdown file containing a detailed report about the specified CVE.

### Advanced Options

The `generate-cve-report` command supports the same options as the standard content generation:

```bash
# Add a custom title
llm-markdown generate-cve-report CVE-2023-12345 --title "Critical Authentication Bypass in Example Service"

# Specify output directory
llm-markdown generate-cve-report CVE-2023-12345 --output-dir "security/advisories"

# Override the provider and model
llm-markdown generate-cve-report CVE-2023-12345 --provider openai --model gpt-4o

# Use a specific API key
llm-markdown generate-cve-report CVE-2023-12345 --provider gemini --api-key "YOUR_API_KEY"

# Enable verbose output and token usage reporting
llm-markdown generate-cve-report CVE-2023-12345 --verbose --usage-report

# Run without making actual API calls (for testing)
llm-markdown generate-cve-report CVE-2023-12345 --dry-run
```

## Report Structure

The generated security advisories follow a standardized structure as defined in the `security_advisory.j2` template:

1. **Executive Summary** - A concise overview of the vulnerability and its significance
2. **Vulnerability Snapshot** - Table containing:
   - CVE ID (hyperlinked to the CVE record)
   - CVSS Score with vector string
   - EPSS Score (probability of exploitation)
   - CWE Category
   - Affected Software/Systems
   - Vulnerability Type (e.g., RCE, Privilege Escalation)
   - Patch Availability
   - Exploitation Status
3. **Technical Details** - In-depth explanation of:
   - Vulnerability description
   - Attack vectors
   - Root cause analysis
   - System/application impact
4. **Threat Actor Activity** - Information about known threat actors exploiting the vulnerability
5. **Impact Assessment** - Analysis of:
   - Business impact
   - Operational impact
   - Data security implications
6. **Mitigation and Remediation** - Detailed steps including:
   - Official patches and updates
   - Workarounds
   - Configuration changes
   - Detection methods
7. **Conclusion** - Summary of key actions
8. **References** - Links to official advisories and resources

## Customizing Reports

To customize the format and content of security advisories, you can modify the `security_advisory.j2` template located in the `.llmconfig/prompt-templates/` directory.

## Example Report

Here's an example of what a generated report looks like:

```markdown
# Security Advisory: CVE-2023-12345

## Executive Summary

CVE-2023-12345 is a critical remote code execution vulnerability affecting Apache Struts versions 2.0.0 to 2.5.30. 
This vulnerability allows attackers to execute arbitrary code on affected systems by sending specially crafted 
requests to the vulnerable application. With a CVSS score of 9.8 (Critical) and a high EPSS score of 0.78, 
indicating a 78% probability of exploitation in the wild, immediate patching is strongly recommended.

## Vulnerability Snapshot

| CVE ID | CVSS Score | EPSS Score | Affected Systems | Patch Status |
|--------|------------|------------|------------------|--------------|
| [CVE-2023-12345](https://www.cve.org/CVERecord?id=CVE-2023-12345) | 9.8 Critical [CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H](https://www.first.org/cvss/calculator/3.1#CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H) | 0.78 (High) | Apache Struts 2.0.0-2.5.30 | [Available](https://struts.apache.org/security) |

## Technical Details

...
```

## Programmatic Usage

You can also use the security report generation feature in your Python code:

```python
from llm_markdown_generator.config import load_config, TopicConfig
from llm_markdown_generator.llm_provider import OpenAIProvider
from llm_markdown_generator.prompt_engine import PromptEngine
from llm_markdown_generator.front_matter import FrontMatterGenerator
from llm_markdown_generator.generator import MarkdownGenerator

# Example of generating a CVE report programmatically
def generate_cve_report(cve_id, output_dir=None):
    # Load configuration
    config = load_config("config/config.yaml")
    
    # Override output directory if provided
    if output_dir:
        config.output_dir = output_dir
        
    # Set up LLM provider
    llm_provider = OpenAIProvider(
        model_name="gpt-4",
        api_key_env_var="OPENAI_API_KEY",
        temperature=0.7
    )
    
    # Set up prompt engine and front matter generator
    prompt_engine = PromptEngine(".llmconfig/prompt-templates")
    front_matter_generator = FrontMatterGenerator("config/front_matter_schema.yaml")
    
    # Create generator
    generator = MarkdownGenerator(
        config=config,
        llm_provider=llm_provider,
        prompt_engine=prompt_engine,
        front_matter_generator=front_matter_generator,
    )
    
    # Add CVE topic to config
    topic_name = "security_advisory"
    config.topics[topic_name] = TopicConfig(
        name=topic_name,
        prompt_template="security_advisory.j2",
        keywords=["security", "vulnerability", "CVE", cve_id],
        custom_data={}
    )
    
    # Prepare custom parameters
    custom_params = {
        "topic": cve_id,
        "title": f"Security Advisory: {cve_id}",
        "audience": "security professionals and IT administrators",
        "keywords": ["cybersecurity", "vulnerability", "CVSS", "EPSS", cve_id],
    }
    
    # Generate content
    content = generator.generate_content(topic_name, custom_params)
    
    # Write to file
    output_path = generator.write_to_file(content, title=custom_params["title"])
    
    return output_path

# Example usage
if __name__ == "__main__":
    report_path = generate_cve_report("CVE-2023-12345", "security_reports")
    print(f"Report generated at: {report_path}")
```

## Integration Examples

The repository includes example scripts in the `examples/` directory:

- `generate_cve_report.py`: Demonstrates how to use the CVE report generation feature
- `generate_security_report.py`: A more comprehensive example for security content generation

## Further Reading

- [Security Templates](../llmconfig/prompt-templates/security_advisory.j2) - The template used for generating security advisories
- [CVE Database](https://www.cve.org/) - Official source for CVE vulnerability information
- [CVSS Calculator](https://www.first.org/cvss/calculator/3.1) - For understanding CVSS scores
- [EPSS](https://www.first.org/epss/) - For understanding Exploit Prediction Scoring System