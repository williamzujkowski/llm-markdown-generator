"""
Example script demonstrating how to generate a CVE report using the CLI.

This example shows how to use the `generate_cve_report` command from the CLI to create
a comprehensive security advisory for a specific CVE ID.
"""

import subprocess
import sys
import os
from pathlib import Path


def main():
    """Run the CVE report generation example."""
    # Get the CVE ID from command line arguments or use a default
    cve_id = sys.argv[1] if len(sys.argv) > 1 else "CVE-2023-12345"
    
    # Use the CLI command to generate the report
    print(f"Generating security advisory for {cve_id}...")
    
    # Construct the command
    # We're using --dry-run to avoid making actual API calls in the example
    cmd = [
        "python", "-m", "llm_markdown_generator.cli", "generate-cve-report",
        cve_id,
        "--output-dir", "output/security",
        "--dry-run",
        "--verbose"
    ]
    
    # Create the output directory if it doesn't exist
    os.makedirs("output/security", exist_ok=True)
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print("\nCommand completed successfully.")
        print("In a real run (without --dry-run), this would generate a comprehensive")
        print("security advisory using the security_advisory.j2 template in .llmconfig/prompt-templates/")
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return
    
    # Show how to run with other options
    print("\nOther example command options:")
    print("------------------------------")
    print(f"python -m llm_markdown_generator.cli generate-cve-report {cve_id} --provider openai --model gpt-4o")
    print(f"python -m llm_markdown_generator.cli generate-cve-report {cve_id} --title 'Custom CVE Report Title'")
    print("python -m llm_markdown_generator.cli generate-cve-report --help")


if __name__ == "__main__":
    main()