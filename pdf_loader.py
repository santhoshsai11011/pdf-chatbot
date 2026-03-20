"""
Service Layer — PDF Extraction with OCR Fallback

Responsibility: Extract text content from PDF files page-by-page.
Uses PyMuPDF as primary extractor with pytesseract OCR fallback
for scanned pages (those yielding fewer than 50 characters).

Permitted imports: Python stdlib, PyMuPDF (fitz), pytesseract, PIL,
    infrastructure layer (config, logger, exceptions).
Must NOT import: streamlit, rag_pipeline, app, or any UI/pipeline module.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, List, Union

import fitz  # PyMuPDF

from exceptions import PDFLoadError
from logger import get_logger

logger = get_logger(__name__)

# Minimum characters on a page before triggering OCR fallback
_OCR_CHAR_THRESHOLD = 50


@dataclass
class PageContent:
    """Represents extracted text from a single PDF page.

    Attributes:
        page: Zero-indexed page number.
        text: Extracted text content.
        source: Original filename or path.
        ocr_used: Whether OCR was used for this page.
    """

    page: int
    text: str
    source: str
    ocr_used: bool


def load_pdf(
    file_input: Union[str, Path, BinaryIO],
    source_name: str = "unknown.pdf",
) -> List[PageContent]:
    """Extract text from a PDF file, with OCR fallback for scanned pages.

    Args:
        file_input: File path (str/Path) or binary file-like object.
        source_name: Human-readable name for logging and source tracking.

    Returns:
        List of PageContent, one per page with non-empty text.

    Raises:
        PDFLoadError: If the PDF cannot be opened or parsed.
    """
    try:
        if isinstance(file_input, (str, Path)):
            source_name = str(file_input)
            doc = fitz.open(str(file_input))
        else:
            # Read bytes from file-like object
            pdf_bytes = file_input.read()
            if not pdf_bytes:
                raise PDFLoadError(source_name, "Empty file")
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except PDFLoadError:
        raise
    except Exception as e:
        raise PDFLoadError(source_name, str(e)) from e

    if doc.page_count == 0:
        raise PDFLoadError(source_name, "PDF has zero pages")

    pages: List[PageContent] = []
    logger.info(
        "Processing PDF '%s' with %d pages", source_name, doc.page_count
    )

    for page_num in range(doc.page_count):
        try:
            page = doc.load_page(page_num)
            text = page.get_text("text").strip()
            ocr_used = False

            # OCR fallback for pages with very little text
            if len(text) < _OCR_CHAR_THRESHOLD:
                ocr_text = _ocr_page(page, page_num, source_name)
                if ocr_text:
                    text = ocr_text
                    ocr_used = True

            if text:
                pages.append(
                    PageContent(
                        page=page_num,
                        text=text,
                        source=source_name,
                        ocr_used=ocr_used,
                    )
                )
        except Exception as e:
            logger.warning(
                "Failed to extract page %d from '%s': %s",
                page_num,
                source_name,
                e,
            )
            continue

    doc.close()

    if not pages:
        raise PDFLoadError(source_name, "No text could be extracted from any page")

    logger.info(
        "Extracted %d pages from '%s' (%d used OCR)",
        len(pages),
        source_name,
        sum(1 for p in pages if p.ocr_used),
    )
    return pages


def _ocr_page(page: fitz.Page, page_num: int, source_name: str) -> str:
    """Attempt OCR on a page using pytesseract.

    Args:
        page: PyMuPDF page object.
        page_num: Page number for logging.
        source_name: PDF source name for logging.

    Returns:
        Extracted OCR text, or empty string if OCR fails or is unavailable.
    """
    try:
        import pytesseract
        from PIL import Image
        import io

        logger.warning(
            "Page %d of '%s' has < %d chars — attempting OCR",
            page_num,
            source_name,
            _OCR_CHAR_THRESHOLD,
        )

        # Render page to image at 300 DPI for good OCR quality
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))

        text = pytesseract.image_to_string(image).strip()
        if text:
            logger.info(
                "OCR extracted %d chars from page %d of '%s'",
                len(text),
                page_num,
                source_name,
            )
        return text

    except ImportError:
        logger.warning(
            "pytesseract not available — skipping OCR for page %d of '%s'",
            page_num,
            source_name,
        )
        return ""
    except Exception as e:
        logger.warning(
            "OCR failed for page %d of '%s': %s",
            page_num,
            source_name,
            e,
        )
        return ""
