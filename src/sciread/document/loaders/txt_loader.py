"""Text file loader implementation."""

from pathlib import Path

import chardet

from .base import BaseLoader
from .base import LoadResult


class TxtLoader(BaseLoader):
    """Loader for plain text files."""

    @property
    def supported_extensions(self) -> list[str]:
        """Return supported file extensions."""
        return [".txt", ".text", ".md", ".rst"]

    @property
    def loader_name(self) -> str:
        """Return the loader name."""
        return "TxtLoader"

    def load(self, file_path: Path) -> LoadResult:
        """Load text content from a plain text file."""
        result = LoadResult(text="", metadata=self._create_metadata(file_path))

        try:
            # Try to detect encoding
            encoding = self._detect_encoding(file_path)
            result.extraction_info["detected_encoding"] = encoding

            # Read file with detected encoding
            with file_path.open(encoding=encoding) as f:
                text = f.read()

            if not text.strip():
                result.add_warning("File is empty or contains only whitespace")
                result.text = ""
            else:
                result.text = text

            # Add text statistics
            result.extraction_info.update(
                {
                    "character_count": len(text),
                    "word_count": len(text.split()),
                    "line_count": len(text.splitlines()),
                }
            )

        except UnicodeDecodeError as e:
            result.add_error(f"Failed to decode file: {e}")
        except OSError as e:
            result.add_error(f"Failed to read file: {e}")
        except Exception as e:
            result.add_error(f"Unexpected error loading file: {e}")

        return result

    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding using chardet."""
        try:
            with file_path.open("rb") as f:
                raw_data = f.read(10240)  # Read first 10KB for detection
                if not raw_data:
                    return "utf-8"  # Default for empty files

                result = chardet.detect(raw_data)
                encoding = result.get("encoding", "utf-8")

                # Validate encoding
                if encoding:
                    try:
                        raw_data.decode(encoding)
                        return encoding
                    except (UnicodeDecodeError, LookupError):
                        pass

                # Fallback to common encodings
                for fallback in ["utf-8", "latin-1", "ascii"]:
                    try:
                        raw_data.decode(fallback)
                        return fallback
                    except (UnicodeDecodeError, LookupError):
                        continue

                # Last resort
                return "utf-8"

        except Exception:
            return "utf-8"
