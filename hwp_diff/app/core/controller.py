"""
Core controller: orchestrates loading, comparing, and exporting.
Designed to be testable without UI (uses callback functions for progress).
"""
import os
from pathlib import Path
from typing import Optional, Callable, List

from app.models.document import DocumentStructure
from app.models.change_record import CompareResult
from app.normalizers.document_normalizer import DocumentNormalizer
from app.diff_engine.diff_engine import DiffEngine
from app.exporters.excel_exporter import ExcelExporter
from app.utils.logger import get_logger

logger = get_logger("core.controller")

SUPPORTED_EXTENSIONS = {
    ".hwpx": "hwpx",
    ".hwp": "hwp",
    ".docx": "docx",
    ".txt": "txt",
}


class CompareController:
    """
    Main application controller.

    Responsibilities:
    - Detect document format and dispatch to correct parser
    - Normalize parsed documents
    - Run comparison via DiffEngine
    - Export results via ExcelExporter
    - Report progress via callbacks
    """

    def __init__(self):
        self._normalizer = DocumentNormalizer()
        self._progress_callback: Optional[Callable[[int, str], None]] = None
        self._last_result: Optional[CompareResult] = None
        self._old_doc: Optional[DocumentStructure] = None
        self._new_doc: Optional[DocumentStructure] = None

    def set_progress_callback(self, cb: Callable[[int, str], None]) -> None:
        """Set a progress callback: cb(percent: int, message: str)."""
        self._progress_callback = cb

    def _progress(self, pct: int, msg: str) -> None:
        logger.info("[%d%%] %s", pct, msg)
        if self._progress_callback:
            self._progress_callback(pct, msg)

    # ------------------------------------------------------------------
    # Document loading
    # ------------------------------------------------------------------

    def load_document(
        self,
        path: str,
        options: Optional[dict] = None,
    ) -> DocumentStructure:
        """
        Load and parse a document from the given file path.

        Args:
            path: Absolute path to the document
            options: Normalization options (ignore_whitespace, etc.)

        Returns:
            Normalized DocumentStructure

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If format is not supported
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

        suffix = p.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"지원하지 않는 파일 형식입니다: {suffix}\n"
                f"지원 형식: {', '.join(SUPPORTED_EXTENSIONS.keys())}"
            )

        fmt = SUPPORTED_EXTENSIONS[suffix]
        logger.info("Loading %s as %s", path, fmt)

        parser = self._get_parser(fmt)
        doc = parser.parse(path)

        # Normalize
        doc = self._normalizer.normalize(doc, options or {})

        return doc

    def _get_parser(self, fmt: str):
        """Get the appropriate parser for a file format."""
        if fmt == "hwpx":
            from app.parsers.hwpx_parser import HWPXParser
            return HWPXParser()
        elif fmt == "hwp":
            from app.parsers.hwp_converter import HWPConverter
            return HWPConverter()
        elif fmt == "docx":
            from app.parsers.docx_parser import DOCXParser
            return DOCXParser()
        elif fmt == "txt":
            from app.parsers.txt_parser import TXTParser
            return TXTParser()
        else:
            raise ValueError(f"Unknown format: {fmt}")

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def run_compare(
        self,
        old_doc: DocumentStructure,
        new_doc: DocumentStructure,
        options: Optional[dict] = None,
    ) -> CompareResult:
        """
        Run comparison between two documents.

        Args:
            old_doc: Original document structure
            new_doc: New document structure
            options: DiffEngine options dict

        Returns:
            CompareResult with all detected changes
        """
        self._old_doc = old_doc
        self._new_doc = new_doc

        engine = DiffEngine(options or {})
        engine.set_progress_callback(self._progress)

        self._progress(0, "비교 시작...")
        result = engine.compare(old_doc, new_doc)
        self._last_result = result
        self._progress(100, f"비교 완료: 총 {len(result.changes)}건 변경")
        return result

    def run_compare_from_paths(
        self,
        old_path: str,
        new_path: str,
        options: Optional[dict] = None,
    ) -> CompareResult:
        """
        Convenience method: load both documents and compare.

        Args:
            old_path: Path to original document
            new_path: Path to new document
            options: Combined options for loading and comparison

        Returns:
            CompareResult
        """
        opts = options or {}

        self._progress(5, f"기준문서 로드: {Path(old_path).name}")
        old_doc = self.load_document(old_path, opts)

        self._progress(20, f"비교문서 로드: {Path(new_path).name}")
        new_doc = self.load_document(new_path, opts)

        self._progress(30, "비교 시작...")
        result = self.run_compare(old_doc, new_doc, opts)
        return result

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_excel(
        self,
        result: Optional[CompareResult] = None,
        output_path: str = "",
        old_file_name: str = "",
        new_file_name: str = "",
        product_name: str = "",
        spec_number: str = "",
        template_path: str = "",
    ) -> bool:
        """
        Export comparison result to Excel.

        Args:
            result: CompareResult to export (uses last result if None)
            output_path: Output file path (.xlsx)
            old_file_name: Display name for original doc
            new_file_name: Display name for new doc
            product_name: 품명 to fill in Excel B column
            spec_number: 규격서번호/도번 to fill in Excel C column
            template_path: Path to QAR template .xlsx (F:\\template2.xlsx)

        Returns:
            True on success
        """
        if result is None:
            result = self._last_result
        if result is None:
            logger.error("No comparison result to export")
            return False

        if not output_path:
            logger.error("No output path specified")
            return False

        # Derive display names from stored docs if not provided
        if not old_file_name and self._old_doc:
            old_file_name = Path(self._old_doc.file_path).name
        if not new_file_name and self._new_doc:
            new_file_name = Path(self._new_doc.file_path).name

        old_file_name = old_file_name or "기준문서"
        new_file_name = new_file_name or "비교문서"

        # Compute total pages from stored documents
        total_pages = 0
        if self._new_doc and self._new_doc.total_pages > 0:
            total_pages = self._new_doc.total_pages
        elif self._old_doc and self._old_doc.total_pages > 0:
            total_pages = self._old_doc.total_pages

        exporter = ExcelExporter()
        success = exporter.export(
            result,
            output_path,
            old_file_name,
            new_file_name,
            product_name=product_name,
            spec_number=spec_number,
            template_path=template_path,
            total_pages=total_pages,
        )

        if success:
            logger.info("Excel export success: %s", output_path)
        else:
            logger.error("Excel export failed: %s", output_path)

        return success

    # ------------------------------------------------------------------
    # Metadata / utility
    # ------------------------------------------------------------------

    def get_supported_formats(self) -> List[str]:
        """Return list of supported file extensions."""
        return list(SUPPORTED_EXTENSIONS.keys())

    def get_last_result(self) -> Optional[CompareResult]:
        """Return the most recent comparison result."""
        return self._last_result

    def get_loaded_documents(self):
        """Return (old_doc, new_doc) tuple."""
        return self._old_doc, self._new_doc

    def clear(self) -> None:
        """Reset state."""
        self._last_result = None
        self._old_doc = None
        self._new_doc = None
