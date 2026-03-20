"""Tests for the PDF loader service."""

import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from exceptions import PDFLoadError
from pdf_loader import PageContent, load_pdf


class TestLoadPdf:
    """Tests for the load_pdf function."""

    def test_valid_pdf_extraction(self, sample_pdf_file):
        """Test that a valid PDF returns non-empty PageContent list."""
        pages = load_pdf(sample_pdf_file, source_name="test.pdf")

        assert len(pages) > 0
        assert all(isinstance(p, PageContent) for p in pages)
        assert pages[0].source == "test.pdf"
        assert pages[0].page == 0
        assert len(pages[0].text) > 0

    def test_page_content_fields(self, sample_pdf_file):
        """Test that PageContent has all required fields populated."""
        pages = load_pdf(sample_pdf_file, source_name="test.pdf")

        page = pages[0]
        assert isinstance(page.page, int)
        assert isinstance(page.text, str)
        assert isinstance(page.source, str)
        assert isinstance(page.ocr_used, bool)
        assert page.ocr_used is False  # Normal text PDF, no OCR needed

    def test_empty_pdf_raises_error(self, empty_pdf_bytes):
        """Test that a PDF with no extractable text raises PDFLoadError."""
        file_obj = io.BytesIO(empty_pdf_bytes)

        with pytest.raises(PDFLoadError):
            load_pdf(file_obj, source_name="empty.pdf")

    def test_corrupt_input_raises_error(self):
        """Test that corrupt/invalid data raises PDFLoadError."""
        corrupt_data = io.BytesIO(b"this is not a pdf file at all")

        with pytest.raises(PDFLoadError) as exc_info:
            load_pdf(corrupt_data, source_name="corrupt.pdf")

        assert "corrupt.pdf" in str(exc_info.value)

    def test_empty_bytes_raises_error(self):
        """Test that empty bytes raise PDFLoadError."""
        empty_file = io.BytesIO(b"")

        with pytest.raises(PDFLoadError):
            load_pdf(empty_file, source_name="empty_bytes.pdf")

    def test_source_name_propagated(self, sample_pdf_file):
        """Test that the source_name is correctly propagated to pages."""
        pages = load_pdf(sample_pdf_file, source_name="my_report.pdf")

        for page in pages:
            assert page.source == "my_report.pdf"
