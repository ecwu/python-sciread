"""Text processing utilities for document analysis.

This module provides helper functions for processing document text,
including removing reference sections and cleaning up academic paper content.
"""

import re


def remove_references(text: str, reference_keywords: list[str] | None = None) -> str:
    """Remove text after the reference section in academic papers.

    This function attempts to identify where the reference section begins
    and removes everything after it, including lengthy appendices and citations
    that might interfere with document analysis.

    Args:
        text: The full text content of the document
        reference_keywords: List of keywords that might indicate the start of references.
                          Defaults to common academic reference section headers.

    Returns:
        Text with reference section and subsequent content removed
    """
    if not text or not text.strip():
        return text

    # Default reference section keywords (case-insensitive)
    default_keywords = [
        "references",
        "bibliography",
        "cited works",
        "works cited",
        "reference",
        "appendix",
        "acknowledgments",
        "acknowledgement",
    ]

    keywords = reference_keywords or default_keywords

    # Create a regex pattern that matches any of the keywords
    # Look for patterns like:
    # References\n
    # REFERENCES\n
    # References\n\n
    # References: or References.
    pattern = r"(?i)^\s*(" + "|".join(re.escape(keyword) for keyword in keywords) + r")\s*[:.\-]?\s*$"

    lines = text.split("\n")

    for i, line in enumerate(lines):
        if re.match(pattern, line.strip()):
            # Found the start of reference section
            # Check if there's substantial content after this section
            remaining_lines = lines[i + 1 :]

            # Count characters in the remaining content
            remaining_text = "\n".join(remaining_lines)
            remaining_chars = len(remaining_text.strip())

            # If there's a lot of content after references (likely appendix/citations),
            # remove it. Keep only up to a reasonable amount (e.g., 2000 chars)
            if remaining_chars > 2000:
                # Return text up to the reference section
                return "\n".join(lines[:i]).rstrip()
            else:
                # Keep the references and everything after if it's not too long
                break

    return text


def clean_academic_text(text: str) -> str:
    """Clean academic paper text for better processing.

    This function performs basic text cleaning operations to improve
    the quality of text processing for academic papers.

    Args:
        text: Raw text content

    Returns:
        Cleaned text content
    """
    if not text or not text.strip():
        return text

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Fix common PDF extraction issues
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)  # Add space between camelCase words
    text = re.sub(r"(\w)(\d)", r"\1 \2", text)  # Space between letters and numbers
    text = re.sub(r"(\d)(\w)", r"\1 \2", text)  # Space between numbers and letters

    # Remove page numbers and headers/footers patterns
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        # Skip likely page numbers (standalone numbers with few digits)
        if re.match(r"^\d{1,3}$", line):
            continue

        # Skip likely headers/footers (short lines with common patterns)
        if len(line) < 50 and re.search(r"\d{1,3}\s*(of|/)\s*\d{1,3}", line):
            continue

        # Skip lines that are just URLs or DOIs
        if re.match(r"^(https?://|doi:|www\.)", line):
            continue

        cleaned_lines.append(line)

    # Rejoin and clean up spacing
    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)  # Reduce excessive newlines
    cleaned_text = cleaned_text.strip()

    return cleaned_text


def extract_document_metadata(text: str) -> dict[str, str]:
    """Extract basic metadata from academic paper text.

    Args:
        text: Document text content

    Returns:
        Dictionary containing extracted metadata
    """
    metadata = {}
    lines = text.split("\n")[:50]  # Check first 50 lines for metadata

    # Try to find title (usually first or second non-empty line, often all caps or title case)
    for _i, line in enumerate(lines):
        line = line.strip()
        if len(line) > 10 and len(line) < 200:  # Reasonable title length
            # Skip if it looks like an author line (contains "University", "Institute", etc.)
            if any(word in line.lower() for word in ["university", "institute", "department", "college", "lab", "laboratory"]):
                continue
            metadata["title"] = line
            break

    # Try to find authors (look for lines with names after title)
    for _i, line in enumerate(lines[1:20], 1):  # Check lines 2-20
        line = line.strip()
        if line and len(line) < 100:  # Reasonable author line length
            # Common author patterns
            if re.search(r"^[A-Z][a-z]+ [A-Z][a-z]+", line) and "university" not in line.lower():
                metadata["authors"] = line
                break

    # Try to find abstract
    abstract_started = False
    abstract_lines = []

    for line in lines:
        line_lower = line.lower().strip()

        if line_lower.startswith("abstract"):
            abstract_started = True
            continue

        if abstract_started:
            if line_lower and not line_lower.startswith(("introduction", "keywords", "1.", "i.", "©")):
                abstract_lines.append(line.strip())
            else:
                break

    if abstract_lines:
        metadata["abstract"] = " ".join(abstract_lines)

    return metadata
