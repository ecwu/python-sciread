"""Base interface for document loaders."""

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from sciread.document.models import DocumentMetadata


@dataclass
class LoadResult:
    """Result of loading a document."""

    text: str
    metadata: DocumentMetadata
    success: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    extraction_info: dict[str, Any] = field(default_factory=dict)

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.success = False

    @property
    def has_issues(self) -> bool:
        """Check if there are any warnings or errors."""
        return bool(self.warnings or self.errors)


class BaseLoader(ABC):
    """Abstract base class for document loaders."""

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Return list of supported file extensions."""

    @property
    @abstractmethod
    def loader_name(self) -> str:
        """Return the name of this loader."""

    @abstractmethod
    def load(self, file_path: Path) -> LoadResult:
        """Load text and metadata from a file."""

    def can_load(self, file_path: Path) -> bool:
        """Check if this loader can handle the given file."""
        if not file_path.exists() or not file_path.is_file():
            return False

        return file_path.suffix.lower() in self.supported_extensions

    def _create_metadata(self, file_path: Path) -> DocumentMetadata:
        """Create basic metadata from file path."""
        try:
            stat = file_path.stat()
            return DocumentMetadata(
                source_path=file_path,
                file_type=file_path.suffix.lower().lstrip("."),
                file_size=stat.st_size,
                modified_at=stat.st_mtime,
            )
        except OSError:
            return DocumentMetadata(
                source_path=file_path,
                file_type=file_path.suffix.lower().lstrip("."),
            )
