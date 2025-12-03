"""
OCR-based URL extraction from screenshots.

This module provides functionality to extract URLs from images
using OCR (Optical Character Recognition).

Supports various URL formats including:
- Full URLs: https://example.com, http://example.com
- Domain-only URLs: google.com, github.com/user/repo
- Subdomains: docs.google.com, api.github.com
"""

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


class OCRUrlExtractor:
    """Extract URLs from images using OCR"""

    # Regex patterns for URL extraction
    # Full URL pattern (http/https)
    FULL_URL_PATTERN = re.compile(
        r'https?://[^\s<>"{}|\\^`\[\]]+',
        re.IGNORECASE
    )

    # Domain pattern (without protocol) - matches google.com, github.com/path, etc.
    # More conservative pattern to avoid false positives
    DOMAIN_PATTERN = re.compile(
        r'\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}(?:/[^\s<>"{}|\\^`\[\]]*)?',
        re.IGNORECASE
    )

    @staticmethod
    def extract_urls_from_image(image_path: str) -> List[str]:
        """
        Extract URLs from an image using OCR.

        Supports multiple URL formats:
        - https://example.com
        - http://example.com
        - google.com
        - github.com/user/repo
        - docs.google.com

        Args:
            image_path: Path to the image file

        Returns:
            List of extracted and normalized URLs (with https:// prefix added if needed)
        """
        try:
            # Try to import OCR library
            try:
                import pytesseract
                from PIL import Image
            except ImportError:
                logger.warning(
                    "pytesseract or PIL not available, skipping OCR URL extraction"
                )
                return []

            # Check if tesseract is installed
            try:
                pytesseract.get_tesseract_version()
            except Exception:
                logger.warning("Tesseract OCR not installed, skipping OCR URL extraction")
                return []

            # Perform OCR with optimized settings
            image = Image.open(image_path)

            # Configure OCR with multiple languages and optimized settings
            # Languages: English + Simplified Chinese + Traditional Chinese
            # PSM 6: Assume a single uniform block of text
            # OEM 3: Default OCR Engine mode (best for most cases)
            ocr_config = r'--psm 6 --oem 3'

            # Try multi-language OCR first (eng+chi_sim+chi_tra)
            # Fall back to English-only if language packs not installed
            try:
                text = pytesseract.image_to_string(image, lang='eng+chi_sim+chi_tra', config=ocr_config)
                logger.debug(f"OCR with multi-language (eng+chi_sim+chi_tra)")
            except pytesseract.pytesseract.TesseractError as lang_error:
                # If multi-language fails, try English + Simplified Chinese
                try:
                    text = pytesseract.image_to_string(image, lang='eng+chi_sim', config=ocr_config)
                    logger.debug(f"OCR with eng+chi_sim")
                except pytesseract.pytesseract.TesseractError:
                    # Fall back to English only
                    text = pytesseract.image_to_string(image, lang='eng', config=ocr_config)
                    logger.warning(f"OCR fallback to English-only (Chinese language packs not installed)")

            logger.debug(f"OCR extracted text from {image_path}: {text[:100]}...")

            # Extract URLs using regex patterns
            full_urls = OCRUrlExtractor.FULL_URL_PATTERN.findall(text)
            domains = OCRUrlExtractor.DOMAIN_PATTERN.findall(text)

            # Normalize URLs
            normalized_urls = []

            # Add full URLs as-is
            normalized_urls.extend(full_urls)

            # Normalize domain-only URLs (add https://)
            for domain in domains:
                # Skip if already in full_urls
                if any(domain in url for url in full_urls):
                    continue

                # Filter out common false positives
                if OCRUrlExtractor._is_likely_url(domain):
                    # Add https:// prefix if not present
                    normalized_url = f"https://{domain}"
                    normalized_urls.append(normalized_url)

            # Remove duplicates while preserving order
            unique_urls = list(dict.fromkeys(normalized_urls))

            if unique_urls:
                logger.info(f"🔍 OCR extracted {len(unique_urls)} URLs from screenshot")
                for url in unique_urls:
                    logger.debug(f"  - {url}")

            return unique_urls

        except Exception as e:
            logger.error(f"Error extracting URLs from image via OCR: {e}")
            return []

    @staticmethod
    def _is_likely_url(domain: str) -> bool:
        """
        Filter out common false positives in domain detection.

        Args:
            domain: The domain string to validate

        Returns:
            True if likely to be a valid URL, False otherwise
        """
        # Filter out strings that are too short
        if len(domain) < 4:
            return False

        # Filter out strings with no TLD
        if '.' not in domain:
            return False

        # Filter out common false positives
        false_positives = [
            'etc.',
            'e.g.',
            'i.e.',
            'vs.',
            'Dr.',
            'Mr.',
            'Mrs.',
            'Ms.',
            'Inc.',
            'Ltd.',
            'Co.',
        ]

        for fp in false_positives:
            if domain.lower().startswith(fp.lower()):
                return False

        # Check if has valid TLD (at least 2 chars)
        parts = domain.split('.')
        if len(parts) < 2:
            return False

        # Get the TLD (last part before any path)
        tld = parts[-1].split('/')[0]
        if len(tld) < 2 or not tld.isalpha():
            return False

        # Check if domain has at least one alphabetic character
        if not any(c.isalpha() for c in domain):
            return False

        return True

    @staticmethod
    def extract_urls_and_text(image_path: str) -> tuple[str, List[str]]:
        """
        Extract both full OCR text and URLs from an image.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (full_text, list_of_urls)
        """
        try:
            import pytesseract
            from PIL import Image

            image = Image.open(image_path)

            # Use same multi-language configuration as extract_urls_from_image
            ocr_config = r'--psm 6 --oem 3'

            try:
                text = pytesseract.image_to_string(image, lang='eng+chi_sim+chi_tra', config=ocr_config)
            except pytesseract.pytesseract.TesseractError:
                try:
                    text = pytesseract.image_to_string(image, lang='eng+chi_sim', config=ocr_config)
                except pytesseract.pytesseract.TesseractError:
                    text = pytesseract.image_to_string(image, lang='eng', config=ocr_config)

            # Extract URLs from the OCR text
            urls = []
            full_urls = OCRUrlExtractor.FULL_URL_PATTERN.findall(text)
            domains = OCRUrlExtractor.DOMAIN_PATTERN.findall(text)

            # Add full URLs
            urls.extend(full_urls)

            # Add normalized domain URLs
            for domain in domains:
                if any(domain in url for url in full_urls):
                    continue
                if OCRUrlExtractor._is_likely_url(domain):
                    urls.append(f"https://{domain}")

            # Remove duplicates
            urls = list(dict.fromkeys(urls))

            return text, urls

        except Exception as e:
            logger.error(f"Error extracting text and URLs from image: {e}")
            return "", []
