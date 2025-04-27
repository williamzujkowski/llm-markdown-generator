"""Tests for the CVE front matter enhancer plugin."""

import unittest
from pathlib import Path
import sys
import os

# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.llm_markdown_generator.config import load_front_matter_schema
from src.llm_markdown_generator.front_matter import FrontMatterGenerator
from src.llm_markdown_generator.plugins.cve_front_matter_enhancer import (
    cve_front_matter_enhancer,
    extract_cve_id,
    extract_cvss_info,
    extract_epss_score,
    extract_cwe_info,
    extract_vulnerability_type,
    extract_vendor_and_product,
    extract_patch_info,
    extract_exploitation_info,
    extract_description,
)


class TestCVEFrontMatterEnhancer(unittest.TestCase):
    """Test the CVE front matter enhancer plugin."""

    def setUp(self):
        """Set up test fixtures."""
        # Sample markdown content for testing
        self.sample_content = """### CVE-2024-29896: Critical Remote Code Execution Vulnerability in "ExampleCorp SecureFile Transfer Application"

#### Vulnerability Snapshot
- **CVE ID**: [CVE-2024-29896](https://www.cve.org/CVERecord?id=CVE-2024-29896)
- **CVSS Score**: 9.8 ([CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H](https://www.first.org/cvss/calculator/3.1#CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H))
- **CVSS Severity**: Critical
- **EPSS Score**: [0.92](https://epss.cyentia.com/) (92% probability of exploitation)
- **CWE Category**: [CWE-787](https://cwe.mitre.org/data/definitions/787.html) (Out-of-bounds Write)
- **Affected Products**: ExampleCorp SecureFile Transfer Application v3.0 - v3.5 ([Vendor Advisory](https://examplecorp.com/security-advisories/ESA-2024-001))
- **Vulnerability Type**: Remote Code Execution (RCE)
- **Patch Availability**: [Yes](https://examplecorp.com/downloads/securefile-transfer)
- **Exploitation Status**: [PoC Available](https://github.com/security-researcher/CVE-2024-29896-PoC)

#### Technical Details
CVE-2024-29896 is a critical remote code execution vulnerability affecting ExampleCorp's SecureFile Transfer Application. The vulnerability stems from an out-of-bounds write condition within the application's file processing module. Specifically, a specially crafted file name can trigger a buffer overflow, allowing an attacker to overwrite critical memory regions and inject malicious code.

The vulnerability exists due to insufficient bounds checking when parsing filenames provided during file upload requests. An attacker can exploit this flaw by sending a specially crafted filename exceeding the allocated buffer size. This overwrite allows the attacker to control the instruction pointer and execute arbitrary code within the context of the application.
"""

        # Sample front matter data
        self.front_matter_data = {
            "title": "Security Advisory: CVE-2024-29896 - Critical Vulnerability Report",
            "date": "2025-04-26",
            "tags": ["security", "vulnerability", "advisory"],
            "category": "security_advisory",
            "author": "AI Content Generator",
        }

        # Try to load the CVE front matter schema
        try:
            self.schema_path = "config/cve_front_matter_schema.yaml"
            self.schema = load_front_matter_schema(self.schema_path)
        except Exception:
            # Fallback to a minimal schema for testing
            self.schema = {
                "title": "Default Title",
                "date": None,
                "tags": [],
                "author": "AI Content Generator",
            }

    def test_extract_cve_id(self):
        """Test extracting CVE ID from content."""
        cve_id = extract_cve_id(self.sample_content)
        self.assertEqual(cve_id, "CVE-2024-29896")

    def test_extract_cvss_info(self):
        """Test extracting CVSS information from content."""
        cvss_info = extract_cvss_info(self.sample_content)
        self.assertEqual(cvss_info["cvssScore"], 9.8)
        self.assertEqual(cvss_info["cvssVector"], "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H")
        self.assertEqual(cvss_info["cvssSeverity"], "Critical")

    def test_extract_epss_score(self):
        """Test extracting EPSS score from content."""
        epss_score = extract_epss_score(self.sample_content)
        self.assertEqual(epss_score, 0.92)

    def test_extract_cwe_info(self):
        """Test extracting CWE information from content."""
        cwe = extract_cwe_info(self.sample_content)
        self.assertEqual(cwe, "CWE-787")

    def test_extract_vulnerability_type(self):
        """Test extracting vulnerability type from content."""
        vuln_type = extract_vulnerability_type(self.sample_content)
        self.assertEqual(vuln_type, "Remote Code Execution (RCE)")

    def test_extract_vendor_and_product(self):
        """Test extracting vendor and product information from content."""
        vendor_product_info = extract_vendor_and_product(self.sample_content)
        self.assertEqual(vendor_product_info["vendor"], "ExampleCorp")
        self.assertTrue("SecureFile Transfer Application" in vendor_product_info["product"])
        self.assertTrue("ExampleCorp SecureFile Transfer Application v3.0 - v3.5" in vendor_product_info["affectedProductsString"])

    def test_extract_patch_info(self):
        """Test extracting patch information from content."""
        patch_info = extract_patch_info(self.sample_content)
        self.assertTrue(patch_info["patchAvailable"])
        self.assertEqual(patch_info["patchLink"], "https://examplecorp.com/downloads/securefile-transfer")

    def test_extract_exploitation_info(self):
        """Test extracting exploitation information from content."""
        exploitation_info = extract_exploitation_info(self.sample_content)
        self.assertEqual(exploitation_info["exploitationStatus"], "PoC Available")
        self.assertEqual(exploitation_info["exploitationStatusLink"], "https://github.com/security-researcher/CVE-2024-29896-PoC")

    def test_extract_description(self):
        """Test extracting description from content."""
        description = extract_description(self.sample_content)
        self.assertTrue("remote code execution vulnerability" in description.lower())
        self.assertTrue("out-of-bounds write" in description.lower())

    def test_cve_front_matter_enhancer(self):
        """Test the full enhancer function."""
        enhanced_front_matter = cve_front_matter_enhancer(
            front_matter=self.front_matter_data,
            content=self.sample_content,
            topic="CVE-2024-29896"
        )

        # Check that essential fields were added
        self.assertEqual(enhanced_front_matter["cveId"], "CVE-2024-29896")
        self.assertEqual(enhanced_front_matter["cvssScore"], 9.8)
        self.assertEqual(enhanced_front_matter["cvssVector"], "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H")
        self.assertEqual(enhanced_front_matter["cvssSeverity"], "Critical")
        self.assertEqual(enhanced_front_matter["epssScore"], 0.92)
        self.assertEqual(enhanced_front_matter["cwe"], "CWE-787")
        self.assertEqual(enhanced_front_matter["vulnerabilityType"], "Remote Code Execution (RCE)")
        self.assertEqual(enhanced_front_matter["vendor"], "ExampleCorp")
        self.assertTrue("SecureFile Transfer Application" in enhanced_front_matter["product"])
        self.assertTrue(enhanced_front_matter["patchAvailable"])
        self.assertEqual(enhanced_front_matter["exploitationStatus"], "PoC Available")

        # Check that tags were enhanced
        expected_tags = {
            "cybersecurity", 
            "vulnerability", 
            "CVSS", 
            "EPSS", 
            "mitigation", 
            "remediation",
            "CVE-2024-29896",
            "ExampleCorp", 
            "RCE"
        }
        
        # Convert the tags to a set for comparison
        actual_tags = set(enhanced_front_matter["tags"])
        
        # Check that all expected tags are present
        for tag in expected_tags:
            self.assertIn(tag, actual_tags)

    def test_integration_with_generator(self):
        """Test the integration with the FrontMatterGenerator."""
        # Create a FrontMatterGenerator with the CVE schema
        front_matter_generator = FrontMatterGenerator(schema=self.schema)
        
        # Generate front matter with the base data
        base_front_matter = front_matter_generator.generate(self.front_matter_data)
        
        # Verify the base front matter has the expected structure
        self.assertIn("title:", base_front_matter)
        self.assertIn("date:", base_front_matter)
        self.assertIn("tags:", base_front_matter)
        
        # Now enhance the front matter data
        enhanced_data = cve_front_matter_enhancer(
            front_matter=self.front_matter_data,
            content=self.sample_content,
            topic="CVE-2024-29896"
        )
        
        # Generate front matter with the enhanced data
        enhanced_front_matter = front_matter_generator.generate(enhanced_data)
        
        # Verify the enhanced front matter has the new fields
        self.assertIn("cveId:", enhanced_front_matter)
        self.assertIn("cvssScore:", enhanced_front_matter)
        self.assertIn("epssScore:", enhanced_front_matter)
        self.assertIn("vulnerabilityType:", enhanced_front_matter)


if __name__ == "__main__":
    unittest.main()