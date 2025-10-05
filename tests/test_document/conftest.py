"""Test fixtures and configuration for document processing tests."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_txt_file(temp_dir: Path) -> Path:
    """Create a sample text file for testing."""
    content = """Abstract
This is a sample academic paper for testing the document processing functionality.
It contains multiple sections that should be detected by the rule-based splitter.

Introduction
The introduction section provides background information about the research topic.
It explains why this research is important and what gaps it fills in the existing literature.

Methods
This section describes the methodology used in the research. It includes details about
data collection, analysis methods, and experimental procedures. The methods should be
sufficiently detailed to allow replication of the study.

Results
Here we present the findings from our experiments. This section includes statistical
analyses, tables, and figures that support our conclusions. The results are presented
in a clear and systematic manner.

Discussion
In this section, we interpret the results and discuss their implications. We compare
our findings with previous research and address the limitations of our study.

Conclusion
The conclusion summarizes the main findings and contributions of the research.
It also suggests directions for future research and practical applications.
"""
    file_path = temp_dir / "sample_paper.txt"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def sample_academic_text() -> str:
    """Sample academic paper text for testing splitters."""
    return """Abstract
This paper presents a novel approach to document processing using machine learning techniques. We demonstrate significant improvements in accuracy and efficiency compared to existing methods.

1. Introduction
Document processing is a fundamental task in many academic and industrial applications. Traditional methods often struggle with the complexity and variety of modern document formats. Our approach addresses these challenges through a combination of rule-based and machine learning techniques.

1.1. Background
The field of document processing has evolved significantly over the past decades. Early systems relied on manual feature engineering and simple pattern matching rules.

2. Methods
We propose a hybrid approach that combines rule-based splitting with machine learning classification. Our system first applies a set of heuristics to identify document sections, then uses a trained classifier to refine the boundaries.

2.1. Data Collection
We collected a dataset of 10,000 academic papers from various disciplines. The papers span multiple domains including computer science, biology, and social sciences.

2.2. Experimental Setup
Our experiments were conducted on a standard computing platform with 32GB RAM and NVIDIA RTX 3080 GPU.

3. Results
The proposed method achieves 95% accuracy in section identification, a 15% improvement over baseline methods. Processing time is reduced by 40% compared to traditional approaches.

3.1. Quantitative Analysis
Table 1 shows the performance comparison between our method and existing approaches. The improvement is statistically significant (p < 0.001).

4. Discussion
The results demonstrate the effectiveness of our hybrid approach. The combination of rule-based and machine learning methods provides both interpretability and accuracy.

4.1. Limitations
Our method has some limitations. It may not perform well on highly non-standard document formats or papers written in languages other than English.

5. Conclusion
We have presented a novel hybrid approach to document processing that significantly outperforms existing methods. Future work will focus on extending the approach to other languages and document types.

References
1. Smith, J. et al. (2023). "Document Processing: A Survey." Journal of AI Research.
2. Johnson, M. (2022). "Machine Learning for Text Analysis." ACM Computing Surveys.
"""


@pytest.fixture
def empty_text_file(temp_dir: Path) -> Path:
    """Create an empty text file for testing error handling."""
    file_path = temp_dir / "empty.txt"
    file_path.write_text("", encoding="utf-8")
    return file_path


@pytest.fixture
def short_text_file(temp_dir: Path) -> Path:
    """Create a very short text file."""
    content = "This is a very short document."
    file_path = temp_dir / "short.txt"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def corrupted_file(temp_dir: Path) -> Path:
    """Create a corrupted binary file for testing error handling."""
    file_path = temp_dir / "corrupted.bin"
    file_path.write_bytes(b"\x00\x01\x02\x03\x04\x05")
    return file_path
