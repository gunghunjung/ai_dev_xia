"""Abstract base class for all document parsers."""
import abc
import uuid
from pathlib import Path
from typing import Optional

from app.models.document import DocumentStructure
from app.utils.logger import get_logger

logger = get_logger("parsers.base")


class BaseParser(abc.ABC):
    """Abstract base parser that all format-specific parsers must implement."""

    @abc.abstractmethod
    def parse(self, file_path: str) -> DocumentStructure:
        """
        Parse a document file and return a DocumentStructure.

        Args:
            file_path: Absolute path to the document file

        Returns:
            DocumentStructure with all parsed blocks

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file format is not supported
        """
        ...

    def _new_doc_id(self, file_path: str) -> str:
        """Generate a unique document ID based on path and a random suffix."""
        stem = Path(file_path).stem[:20].replace(" ", "_")
        uid = uuid.uuid4().hex[:8]
        return f"{stem}_{uid}"

    def _check_file(self, file_path: str) -> Path:
        """Validate file exists and return Path object."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not p.is_file():
            raise ValueError(f"Not a file: {file_path}")
        return p
