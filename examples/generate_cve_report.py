"""
Example script demonstrating how to generate a CVE report using the enhanced front matter schema.

This example shows how to use the `generate_cve_report` command with the comprehensive
CVE front matter schema, which captures detailed vulnerability information in a structured format.
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

from llm_markdown_generator.config import load_config, load_front_matter_schema
from llm_markdown_generator.llm_provider import OpenAIProvider, GeminiProvider
from llm_markdown_generator.prompt_engine import PromptEngine
from llm_markdown_generator.front_matter import FrontMatterGenerator, slugify
from llm_markdown_generator.generator import MarkdownGenerator


def main():
    """Generate a CVE vulnerability report with enhanced front matter."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate a comprehensive CVE vulnerability report with structured front matter"
    )
    parser.add_argument(
        "cve_id",
        type=str,
        help="CVE ID to generate a report for (e.g., CVE-2024-29896)"
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Custom title for the report (default: auto-generated from CVE ID)"
    )
    parser.add_argument(
        "--severity",
        type=str,
        choices=["Critical", "High", "Medium", "Low"],
        default="Critical",
        help="CVSS severity level"
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "gemini"],
        help="LLM provider to use (overrides config.yaml setting)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to use (overrides config.yaml setting)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/vulnerabilities",
        help="Output directory (default: output/vulnerabilities)"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="config/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--front-matter-schema",
        type=str,
        default="config/cve_front_matter_schema.yaml",
        help="Path to CVE front matter schema file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without making LLM API calls (generates mock content)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    args = parser.parse_args()
    
    # Load main configuration
    config = load_config(args.config_path)
    
    # Load CVE-specific front matter schema
    try:
        cve_schema = load_front_matter_schema(args.front_matter_schema)
        if args.verbose:
            print(f"Loaded CVE front matter schema from {args.front_matter_schema}")
    except Exception as e:
        print(f"Error loading CVE front matter schema: {e}")
        print("Falling back to default front matter schema")
        cve_schema = None
    
    # Create front matter generator with CVE schema or default schema
    front_matter_schema_path = args.front_matter_schema if cve_schema else config.front_matter.schema_path
    front_matter_generator = FrontMatterGenerator(
        schema=cve_schema if cve_schema else load_front_matter_schema(config.front_matter.schema_path)
    )
    
    
    # Create LLM provider (use arguments to override config if provided)
    provider_config = config.llm_provider
    if args.provider:
        provider_config.provider_type = args.provider
    if args.model:
        provider_config.model_name = args.model
        
    if provider_config.provider_type.lower() == "openai":
        llm_client = OpenAIProvider(
            model_name=provider_config.model_name,
            api_key_env_var=provider_config.api_key_env_var,
            temperature=provider_config.temperature,
            max_tokens=provider_config.max_tokens,
            additional_params=provider_config.additional_params
        )
    elif provider_config.provider_type.lower() == "gemini":
        llm_client = GeminiProvider(
            model_name=provider_config.model_name,
            api_key_env_var=provider_config.api_key_env_var,
            temperature=provider_config.temperature,
            max_tokens=provider_config.max_tokens,
            additional_params=provider_config.additional_params
        )
    else:
        raise ValueError(f"Unsupported provider type: {provider_config.provider_type}")
        
    # Create prompt engine with default template location
    templates_dir = ".llmconfig/prompt-templates"  # Default location for templates
    prompt_engine = PromptEngine(templates_dir=templates_dir)
    
    # Create markdown generator
    markdown_generator = MarkdownGenerator(
        config=config,
        llm_provider=llm_client,
        prompt_engine=prompt_engine,
        front_matter_generator=front_matter_generator
    )
    
    # Ensure our CVE front matter enhancer plugin is loaded
    # This will extract details from the content and add them to front matter
    try:
        # Import here to ensure it's registered
        import llm_markdown_generator.plugins.cve_front_matter_enhancer
        
        # Load all available plugins
        plugins_loaded = markdown_generator.load_plugins()
        
        if args.verbose:
            print(f"Loaded plugins: {plugins_loaded}")
            
    except Exception as e:
        print(f"Warning: Could not load CVE enhancer plugin: {e}")
    
    # Prepare default title if not provided
    if not args.title:
        args.title = f"{args.cve_id}: Critical Vulnerability Assessment"
        
    # Prepare front matter with initial values
    # The CVE enhancer plugin will extract and add more fields from the generated content
    front_matter_data = {
        "title": args.title,
        "cveId": args.cve_id,
        "publishDate": datetime.datetime.now().strftime("%Y-%m-%d"),
        "cvssSeverity": args.severity,
        "tags": ["cybersecurity", "vulnerability", "CVE", args.cve_id],
        "author": "AI Content Generator"
    }
    
    # Prepare custom parameters for the security_advisory template
    custom_params = {
        "topic": args.cve_id,  # The CVE ID is used as the topic
        "title": args.title,
        "audience": "security professionals and IT administrators",
        "keywords": [
            "cybersecurity",
            "vulnerability",
            "CVSS",
            "EPSS",
            "mitigation",
            "remediation",
            args.cve_id
        ],
        "front_matter": front_matter_data
    }
    
    # Update the config with security_advisory topic if not present
    if "security_advisory" not in config.topics:
        from llm_markdown_generator.config import TopicConfig
        config.topics["security_advisory"] = TopicConfig(
            name="security_advisory",
            prompt_template="security_advisory.j2",
            keywords=custom_params["keywords"]
        )
    
    # Display prompt if verbose
    if args.verbose:
        print("\n----- PROMPT THAT WOULD BE SENT TO LLM -----")
        rendered_prompt = prompt_engine.render_prompt(
            "security_advisory.j2", 
            {k: v for k, v in custom_params.items() if k != 'front_matter'}
        )
        print(rendered_prompt)
        print("----- END OF PROMPT -----\n")
    
    # Generate content or mock content for dry-run
    if args.dry_run:
        # Create a realistic mock response based on the sample provided
        markdown_content = f"""---
cveId: "{args.cve_id}"
title: "{args.title}"
publishDate: "{front_matter_data['publishDate']}"
cvssScore: 9.8
cvssVector: "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
cvssSeverity: "{args.severity}"
epssScore: 0.92
cwe: "CWE-787"
vulnerabilityType: "Remote Code Execution (RCE)"
vendor: "ExampleCorp"
product: "SecureFile Transfer Application"
affectedProductsString: "ExampleCorp SecureFile Transfer Application v3.0 - v3.5"
patchAvailable: true
patchLink: "https://examplecorp.com/downloads/securefile-transfer"
exploitationStatus: "PoC Available"
exploitationStatusLink: "https://github.com/security-researcher/CVE-2024-29896-PoC"
tags:
  - cybersecurity
  - vulnerability
  - CVSS
  - EPSS
  - mitigation
  - remediation
  - {args.cve_id}
  - ExampleCorp
  - SecureFile Transfer Application
  - RCE
  - Out-of-bounds Write
author: "AI Content Generator"
description: "Critical remote code execution vulnerability ({args.cve_id}, CVSS 9.8, EPSS 92%) in ExampleCorp SecureFile Transfer Application allows unauthenticated attackers complete system compromise via crafted filenames. PoC available."
draft: false
last_modified: null
show_toc: true
---

### {args.cve_id}: Critical Remote Code Execution Vulnerability in "ExampleCorp SecureFile Transfer Application"

#### Vulnerability Snapshot
- **CVE ID**: [{args.cve_id}](https://www.cve.org/CVERecord?id={args.cve_id})
- **CVSS Score**: 9.8 ([CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H](https://www.first.org/cvss/calculator/3.1#CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H))
- **CVSS Severity**: Critical
- **EPSS Score**: [0.92](https://epss.cyentia.com/) (92% probability of exploitation)
- **CWE Category**: [CWE-787](https://cwe.mitre.org/data/definitions/787.html) (Out-of-bounds Write)
- **Affected Products**: ExampleCorp SecureFile Transfer Application v3.0 - v3.5 ([Vendor Advisory](https://examplecorp.com/security-advisories/ESA-2024-001))
- **Vulnerability Type**: Remote Code Execution (RCE)
- **Patch Availability**: [Yes](https://examplecorp.com/downloads/securefile-transfer)
- **Exploitation Status**: [PoC Available](https://github.com/security-researcher/CVE-2024-29896-PoC)


#### Technical Details
{args.cve_id} is a critical remote code execution vulnerability affecting ExampleCorp's SecureFile Transfer Application. The vulnerability stems from an out-of-bounds write condition within the application's file processing module. Specifically, a specially crafted file name can trigger a buffer overflow, allowing an attacker to overwrite critical memory regions and inject malicious code.

The vulnerability exists due to insufficient bounds checking when parsing filenames provided during file upload requests. An attacker can exploit this flaw by sending a specially crafted filename exceeding the allocated buffer size. This overwrite allows the attacker to control the instruction pointer and execute arbitrary code within the context of the application.

#### Exploitation Context
A proof-of-concept exploit for {args.cve_id} has been publicly released and is actively being shared within security communities. While widespread exploitation has not yet been confirmed, the ease of exploitation combined with the availability of a public PoC significantly increases the likelihood of imminent attacks.

Given the high EPSS score of 0.92 and the nature of the vulnerability, active exploitation is expected within 24-48 hours. The vulnerability is remotely exploitable without authentication, making it a prime target for automated attacks. Organizations using the affected versions of ExampleCorp SecureFile Transfer Application are strongly urged to apply available patches immediately.

#### Impact Assessment
Successful exploitation of {args.cve_id} could have severe consequences, including:

- **Complete system compromise:** Attackers can gain full control of the affected server.
- **Data breaches:** Sensitive data transferred through the application could be exfiltrated.
- **Denial of Service:** Attackers could disrupt the availability of the file transfer service.
- **Lateral movement:** Compromised servers can be used as a pivot point for attacks on other internal systems.

The severity of the impact is compounded by the fact that the application is often used to transfer sensitive files, potentially leading to significant data breaches and reputational damage.

#### Mitigation and Remediation
- **Apply patches immediately:** Upgrade to the latest version of ExampleCorp SecureFile Transfer Application (v3.6 or later) available at [https://examplecorp.com/downloads/securefile-transfer](https://examplecorp.com/downloads/securefile-transfer).
- **Workarounds (if patching is not immediately possible):**
    - Disable the affected application if it is not essential.
    - Implement strict network access controls to limit access to the application only to trusted sources.
    - Monitor application logs for suspicious activity.
- **Configuration changes:** None required after patching.
- **Detection methods:**
    - Monitor system logs for unusual process creation or network activity.
    - Implement intrusion detection/prevention systems (IDS/IPS) with signatures designed to detect exploitation attempts. ExampleCorp has released a set of Snort rules for this vulnerability.
    - Analyze network traffic for malicious payloads associated with the exploit.


#### References
- [ExampleCorp Security Advisory ESA-2024-001](https://examplecorp.com/security-advisories/ESA-2024-001)
- [CVE-2024-29896 NVD Entry](https://nvd.nist.gov/vuln/detail/{args.cve_id}) (Placeholder - will be populated when the CVE is officially published)
- [GitHub Repository with PoC Exploit](https://github.com/security-researcher/{args.cve_id}-PoC) (Fictional example)
- [Snort Rules for {args.cve_id}](https://examplecorp.com/security-resources/snort-rules) (Fictional example)
"""
    else:
        # Generate actual content using the LLM
        try:
            markdown_content = markdown_generator.generate_content(
                topic_name="security_advisory",
                custom_params=custom_params
            )
        except Exception as e:
            print(f"Error generating content: {e}")
            print("Falling back to mock content for demonstration purposes")
            # Use a simplified mock response
            markdown_content = f"""---
cveId: "{args.cve_id}"
title: "{args.title}"
publishDate: "{front_matter_data['publishDate']}"
cvssScore: 9.8
cvssVector: "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
cvssSeverity: "{args.severity}"
tags:
  - cybersecurity
  - vulnerability
  - CVE
  - {args.cve_id}
author: "AI Content Generator"
description: "This is a mock response since the LLM API call failed."
---

### {args.cve_id}: Critical Vulnerability Assessment

This is a mock response since the LLM API call failed or timed out. 
In a real run, this would be a comprehensive security advisory for {args.cve_id}.
"""
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename from CVE ID
    filename = f"{args.cve_id.lower()}.md"
    
    # Write content to file
    output_path = output_dir / filename
    with open(output_path, "w") as f:
        f.write(markdown_content)
    
    # Print summary
    print(f"\nGenerated vulnerability report for {args.cve_id}")
    print(f"Output file: {output_path}")
    
    if args.verbose and not args.dry_run:
        print("\nGeneration complete")
            
    print(f"\nNext steps:")
    print(f"  1. Review the generated report at {output_path}")
    print(f"  2. Make any necessary edits or adjustments to the content")
    print(f"  3. Use the enhanced front matter for indexing and filtering in your applications")


if __name__ == "__main__":
    main()