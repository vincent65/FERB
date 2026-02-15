from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DocPage:
    """A single scraped documentation page."""

    filename: str
    url: str
    title: str
    content: str
    word_count: int = 0


class DocsCorpus:
    """Loads scraped markdown documentation and provides it for RLM retrieval.

    Expects a directory containing:
      - ``manifest.json`` with metadata about each page
      - ``markdown/`` subdirectory with ``page_XXXX.md`` files
    """

    def __init__(self, docs_dir: Path) -> None:
        self.docs_dir = docs_dir
        self._pages: list[DocPage] = []
        self._build()

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def _build(self) -> None:
        manifest_path = self.docs_dir / "manifest.json"
        if not manifest_path.exists():
            return

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        markdown_dir = self.docs_dir / "markdown"
        for page_info in manifest.get("pages", []):
            filename = page_info.get("filename", "")
            page_path = markdown_dir / filename
            if not page_path.exists():
                continue

            content = page_path.read_text(encoding="utf-8")
            # Skip empty pages or "Page Not Found" entries.
            if page_info.get("word_count", 0) < 50:
                continue

            self._pages.append(
                DocPage(
                    filename=filename,
                    url=page_info.get("url", ""),
                    title=page_info.get("title", filename),
                    content=content,
                    word_count=page_info.get("word_count", 0),
                )
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def as_context_payload(self) -> dict[str, Any]:
        """Return the full docs corpus formatted for RLM's ``context`` variable.

        The returned dict has a ``query`` key (left empty for the caller to fill)
        and a ``content`` key with the concatenated documentation pages, each
        delimited by a markdown header with the page title and source URL.
        """
        if not self._pages:
            return {"query": "", "content": "No documentation pages available."}

        sections: list[str] = []
        for page in self._pages:
            sections.append(
                f"--- PAGE: {page.title} ---\n"
                f"Source: {page.url}\n\n"
                f"{page.content}"
            )
        return {
            "query": "",
            "content": "\n\n".join(sections),
        }

    def get_page_summaries(self) -> str:
        """One-liner per page, useful for diagnostics or logging."""
        if not self._pages:
            return "No documentation pages loaded."
        lines = []
        for page in self._pages:
            lines.append(f"- [{page.filename}] {page.title} ({page.word_count} words)")
        return "\n".join(lines)

    @property
    def pages(self) -> list[DocPage]:
        return list(self._pages)

    @property
    def total_chars(self) -> int:
        return sum(len(p.content) for p in self._pages)

    def __len__(self) -> int:
        return len(self._pages)

    def __repr__(self) -> str:
        return f"DocsCorpus({self.docs_dir}, {len(self._pages)} pages, {self.total_chars} chars)"
