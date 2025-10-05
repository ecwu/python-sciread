"""Tests for base loader functionality."""

from sciread.document.loaders.base import BaseLoader
from sciread.document.loaders.base import LoadResult
from sciread.document.models import DocumentMetadata


class TestBaseLoader:
    """Test cases for BaseLoader abstract class."""

    def test_load_result_creation(self):
        """Test LoadResult creation and properties."""
        metadata = DocumentMetadata(title="Test")
        result = LoadResult(
            text="Test content",
            metadata=metadata,
            success=True,
            warnings=["Warning 1"],
            errors=["Error 1"],
            extraction_info={"key": "value"},
        )

        assert result.text == "Test content"
        assert result.metadata.title == "Test"
        assert result.success is True
        assert len(result.warnings) == 1
        assert len(result.errors) == 1
        assert result.extraction_info["key"] == "value"
        assert result.has_issues  # Has warnings and errors

    def test_load_result_add_warning(self):
        """Test adding warnings to LoadResult."""
        metadata = DocumentMetadata()
        result = LoadResult(text="test", metadata=metadata, success=True)
        assert len(result.warnings) == 0

        result.add_warning("Test warning")
        assert len(result.warnings) == 1
        assert result.warnings[0] == "Test warning"
        assert result.has_issues

    def test_load_result_add_error(self):
        """Test adding errors to LoadResult."""
        metadata = DocumentMetadata()
        result = LoadResult(text="test", metadata=metadata, success=True)
        assert result.success
        assert len(result.errors) == 0

        result.add_error("Test error")
        assert len(result.errors) == 1
        assert result.errors[0] == "Test error"
        assert not result.success  # Should be marked as failed
        assert result.has_issues

    def test_base_loader_metadata_creation(self, temp_dir):
        """Test BaseLoader metadata creation."""

        class TestLoader(BaseLoader):
            @property
            def supported_extensions(self):
                return [".test"]

            @property
            def loader_name(self):
                return "TestLoader"

            def load(self, file_path):
                return LoadResult(text="", metadata=self._create_metadata(file_path))

        loader = TestLoader()
        test_file = temp_dir / "test.test"
        test_file.write_text("test content")

        metadata = loader._create_metadata(test_file)
        assert metadata.source_path == test_file
        assert metadata.file_type == "test"
        assert metadata.file_size > 0
        assert metadata.modified_at is not None
