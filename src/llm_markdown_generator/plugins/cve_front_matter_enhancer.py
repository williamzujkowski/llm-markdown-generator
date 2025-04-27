"""Front matter enhancer plugin for CVE report generation.

Enhances the front matter for CVE reports by adding comprehensive
vulnerability-specific fields based on the extracted content information.
"""

import re
from typing import Dict, Any, Union, List

from llm_markdown_generator.plugins import plugin_hook


def extract_cve_id(content: str) -> Union[str, None]:
    """Extract CVE ID from content."""
    match = re.search(r'CVE-\d{4}-\d{4,}', content)
    return match.group(0) if match else None


def extract_cvss_info(content: str) -> Dict[str, Any]:
    """Extract CVSS score, vector string, and severity from content."""
    cvss_score_match = re.search(r'CVSS Score.*?([0-9]\.[0-9])', content)
    cvss_vector_match = re.search(r'CVSS:3\.\d/[A-Z:/]+', content)
    severity_match = re.search(r'CVSS Severity.*?(Critical|High|Medium|Low)', content)
    
    return {
        "cvssScore": float(cvss_score_match.group(1)) if cvss_score_match else None,
        "cvssVector": cvss_vector_match.group(0) if cvss_vector_match else None,
        "cvssSeverity": severity_match.group(1) if severity_match else None
    }


def extract_epss_score(content: str) -> Union[float, None]:
    """Extract EPSS score from content."""
    # Look for patterns like "EPSS Score: 0.92" or "EPSS Score: [0.92]"
    epss_match = re.search(r'EPSS Score.*?([0-9]\.[0-9]{1,2})', content)
    if epss_match:
        return float(epss_match.group(1))
    return None


def extract_cwe_info(content: str) -> Union[str, None]:
    """Extract CWE identifier from content."""
    cwe_match = re.search(r'CWE-\d+', content)
    return cwe_match.group(0) if cwe_match else None


def extract_vulnerability_type(content: str) -> Union[str, None]:
    """Extract vulnerability type from content."""
    vuln_type_match = re.search(r'Vulnerability Type.*?([A-Za-z ]+\([A-Z]+\)|[A-Za-z ]+)', content)
    if vuln_type_match:
        return vuln_type_match.group(1).strip()
    return None


def extract_vendor_and_product(content: str) -> Dict[str, str]:
    """Extract vendor and product information from content."""
    # This is a simplified approach - may need more sophisticated parsing
    affected_products_match = re.search(r'Affected Products.*?([^\n]+)', content)
    affected_str = affected_products_match.group(1).strip() if affected_products_match else ""
    
    # Try to extract vendor and product
    vendor = None
    product = None
    
    if affected_str:
        # Common pattern: "VendorName ProductName v1.0-v2.0"
        parts = affected_str.split()
        if len(parts) >= 2:
            # Handle case where vendor is in title but not explicitly in Affected Products
            vendor_match = re.search(r'([A-Za-z]+Corp|[A-Za-z]+ware)\b', content)
            if vendor_match:
                vendor = vendor_match.group(1)
                # Assume product follows vendor in the affected string
                if vendor in affected_str:
                    product_str = affected_str.split(vendor, 1)[1].strip()
                    # Remove version information
                    product = re.sub(r'v\d+\.\d+.*$', '', product_str).strip()
    
    return {
        "vendor": vendor,
        "product": product,
        "affectedProductsString": affected_str
    }


def extract_patch_info(content: str) -> Dict[str, Any]:
    """Extract patch availability and link from content."""
    patch_available = "Yes" in content and "Patch Availability" in content
    patch_link_match = re.search(r'Patch Availability.*?\[(Yes|No)\]\((https?://[^\)]+)\)', content)
    
    return {
        "patchAvailable": patch_available,
        "patchLink": patch_link_match.group(2) if patch_link_match else None
    }


def extract_exploitation_info(content: str) -> Dict[str, Any]:
    """Extract exploitation status and link from content."""
    status_patterns = ["PoC Available", "Active Exploitation", "Theoretical"]
    status = None
    
    for pattern in status_patterns:
        if pattern in content:
            status = pattern
            break
    
    link_match = re.search(r'Exploitation Status.*?\[([^\]]+)\]\((https?://[^\)]+)\)', content)
    
    return {
        "exploitationStatus": status,
        "exploitationStatusLink": link_match.group(2) if link_match else None
    }


def extract_description(content: str) -> str:
    """Generate a concise description summarizing the vulnerability."""
    # Get the first paragraph after "Technical Details" as a description
    match = re.search(r'Technical Details\s*\n(.+?)(?=\n\n|\n####)', content, re.DOTALL)
    if match:
        desc = match.group(1).strip()
        # Limit to a reasonable length
        if len(desc) > 200:
            desc = desc[:197] + "..."
        return desc
    return ""


@plugin_hook('front_matter_enhancer', 'cve_report_enhancer')
def cve_front_matter_enhancer(
    front_matter: Dict[str, Any], 
    content: str,
    topic: str = None,
    **kwargs
) -> Dict[str, Any]:
    """Enhance front matter for CVE reports with comprehensive vulnerability data.
    
    Args:
        front_matter: The original front matter data
        content: The generated markdown content
        topic: The topic name (often the CVE ID)
        **kwargs: Additional parameters
        
    Returns:
        Dict[str, Any]: Enhanced front matter with CVE-specific fields
    """
    # Start with the original front matter
    enhanced_front_matter = front_matter.copy()
    
    # Extract CVE ID (from topic or content)
    cve_id = topic if (topic and topic.startswith("CVE-")) else extract_cve_id(content)
    if cve_id:
        enhanced_front_matter["cveId"] = cve_id
    
    # Extract CVSS information
    cvss_info = extract_cvss_info(content)
    enhanced_front_matter.update(cvss_info)
    
    # Extract EPSS score
    epss_score = extract_epss_score(content)
    if epss_score is not None:
        enhanced_front_matter["epssScore"] = epss_score
    
    # Extract CWE information
    cwe = extract_cwe_info(content)
    if cwe:
        enhanced_front_matter["cwe"] = cwe
    
    # Extract vulnerability type
    vuln_type = extract_vulnerability_type(content)
    if vuln_type:
        enhanced_front_matter["vulnerabilityType"] = vuln_type
    
    # Extract vendor and product information
    vendor_product_info = extract_vendor_and_product(content)
    enhanced_front_matter.update(vendor_product_info)
    
    # Extract patch information
    patch_info = extract_patch_info(content)
    enhanced_front_matter.update(patch_info)
    
    # Extract exploitation information
    exploitation_info = extract_exploitation_info(content)
    enhanced_front_matter.update(exploitation_info)
    
    # Create a meaningful description if not provided
    if not enhanced_front_matter.get("description"):
        enhanced_front_matter["description"] = extract_description(content)
    
    # Update tags with vulnerability-specific ones
    tags = set(enhanced_front_matter.get("tags", []))
    
    # Add standard security tags
    security_tags = [
        "cybersecurity", 
        "vulnerability", 
        "CVSS", 
        "EPSS", 
        "mitigation", 
        "remediation"
    ]
    
    # Add specific tags
    if cve_id:
        security_tags.append(cve_id)
    if vendor_product_info.get("vendor"):
        security_tags.append(vendor_product_info["vendor"])
    if vendor_product_info.get("product"):
        security_tags.append(vendor_product_info["product"])
    if vuln_type:
        # For vulnerability types like "Remote Code Execution (RCE)"
        # Extract the abbreviation if available in parentheses
        if "(" in vuln_type and ")" in vuln_type:
            abbr = vuln_type[vuln_type.find("(")+1:vuln_type.find(")")]
            security_tags.append(abbr)  # Add abbreviation as tag (e.g., "RCE")
        else:
            # If no abbreviation, add the first word
            security_tags.append(vuln_type.split(" ")[0])
    
    # Update tags (convert to list for YAML serialization)
    enhanced_front_matter["tags"] = list(tags.union(security_tags))
    
    # Use current date for publishDate if not set
    if not enhanced_front_matter.get("publishDate") and enhanced_front_matter.get("date"):
        enhanced_front_matter["publishDate"] = enhanced_front_matter["date"]
    
    return enhanced_front_matter