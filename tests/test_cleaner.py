import pytest

from app.ingestion.cleaner import MedicalTextCleaner
from app.ingestion.models import ParsedPage


def test_clean_tabs() -> None:
    cleaner = MedicalTextCleaner()

    pages = [
        ParsedPage(
            page_number=1,
            text="Hello\tWorld",
        )
    ]

    cleaned = cleaner.clean(pages)

    assert cleaned[0].text == "Hello World"


def test_collapse_multiple_spaces() -> None:
    cleaner = MedicalTextCleaner()

    pages = [
        ParsedPage(
            page_number=1,
            text="Hello     World",
        )
    ]

    cleaned = cleaner.clean(pages)

    assert cleaned[0].text == "Hello World"


def test_collapse_blank_lines() -> None:
    cleaner = MedicalTextCleaner()

    pages = [
        ParsedPage(
            page_number=1,
            text="Hello\n\n\n\nWorld",
        )
    ]

    cleaned = cleaner.clean(pages)

    assert cleaned[0].text == "Hello\n\nWorld"


def test_strip_whitespace() -> None:
    cleaner = MedicalTextCleaner()

    pages = [
        ParsedPage(
            page_number=1,
            text="   Hello World   ",
        )
    ]

    cleaned = cleaner.clean(pages)

    assert cleaned[0].text == "Hello World"


def test_empty_input_returns_empty_list() -> None:
    cleaner = MedicalTextCleaner()

    cleaned = cleaner.clean([])

    assert cleaned == []


def test_page_numbers_are_preserved(sample_pages: list[ParsedPage]) -> None:
    cleaner = MedicalTextCleaner()

    cleaned = cleaner.clean(sample_pages)

    assert [page.page_number for page in cleaned] == [1, 2]


def test_original_pages_not_modified(dirty_pages: list[ParsedPage]) -> None:
    cleaner = MedicalTextCleaner()

    original_text = dirty_pages[0].text

    _ = cleaner.clean(dirty_pages)

    assert dirty_pages[0].text == original_text


def test_multiple_pages_are_cleaned(dirty_pages: list[ParsedPage]) -> None:
    cleaner = MedicalTextCleaner()

    cleaned = cleaner.clean(dirty_pages)

    assert cleaned[0].text == "This is page one."
    assert cleaned[1].text == "Another page.\n\nText here."