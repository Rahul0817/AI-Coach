"""Knowledge-base loading and chunking.

Chunking strategy
-----------------
Fixed-size character windows are the common default and the wrong choice for
this corpus. These documents are structured by Markdown headings, and a heading
is a semantic boundary the author already provided — splitting across it mixes
"what the evidence says about supplements" with "when to seek help urgently".

So the splitter works **heading-first**: each ``##`` section becomes a chunk,
and only sections that exceed the size budget are further split on paragraph
boundaries. Every chunk carries its document title and its section heading in
the text itself, so a retrieved fragment is self-describing even out of
context — which matters because the fragment, not the document, is what reaches
the model.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)

KNOWLEDGE_DIR = Path(__file__).resolve().parents[3] / "ml" / "knowledge"

#: Target chunk size in characters. Large enough to hold a complete argument,
#: small enough that five chunks fit comfortably in a prompt budget.
MAX_CHUNK_CHARS = 1400
#: Sections shorter than this are merged into the following one rather than
#: retrieved alone — a two-line chunk is rarely a useful answer.
MIN_CHUNK_CHARS = 220

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)


@dataclass(slots=True)
class DocumentChunk:
    """A retrievable unit of knowledge."""

    id: str
    text: str
    title: str
    section: str
    source: str
    category: str
    attribution: str = ""

    def metadata(self) -> dict[str, str]:
        """Chroma metadata must be flat scalars, hence this explicit shape."""
        return {
            "title": self.title,
            "section": self.section,
            "source": self.source,
            "category": self.category,
            "attribution": self.attribution,
        }


@dataclass(slots=True)
class LoadedDocument:
    path: Path
    frontmatter: dict[str, str]
    body: str
    chunks: list[DocumentChunk] = field(default_factory=list)


def parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    """Split YAML-style frontmatter from the body.

    A full YAML parser is overkill for flat ``key: "value"`` pairs and would add
    a dependency purely for this, so the keys are read directly.
    """
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw

    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, raw[match.end() :]


def _split_long_section(text: str, budget: int) -> list[str]:
    """Split an oversized section on paragraph boundaries, never mid-sentence."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    parts: list[str] = []
    current: list[str] = []
    size = 0

    for paragraph in paragraphs:
        # A single paragraph larger than the budget is emitted alone rather
        # than butchered; retrieval quality beats a tidy size histogram.
        if len(paragraph) > budget and not current:
            parts.append(paragraph)
            continue
        if size + len(paragraph) > budget and current:
            parts.append("\n\n".join(current))
            current, size = [], 0
        current.append(paragraph)
        size += len(paragraph) + 2

    if current:
        parts.append("\n\n".join(current))
    return parts


def chunk_document(document: LoadedDocument) -> list[DocumentChunk]:
    """Produce heading-aware chunks for one document."""
    title = document.frontmatter.get("title", document.path.stem)
    source = document.frontmatter.get("source", "Oviora Knowledge Base")
    category = document.frontmatter.get("category", "general")
    attribution = document.frontmatter.get("attribution", "")

    headings = list(_HEADING_RE.finditer(document.body))
    sections: list[tuple[str, str]] = []

    if not headings:
        sections.append(("Overview", document.body.strip()))
    else:
        # Any preamble before the first heading is kept, not silently dropped.
        preamble = document.body[: headings[0].start()].strip()
        if len(preamble) >= MIN_CHUNK_CHARS:
            sections.append(("Overview", preamble))

        for index, match in enumerate(headings):
            heading = match.group(2).strip()
            start = match.end()
            end = (
                headings[index + 1].start()
                if index + 1 < len(headings)
                else len(document.body)
            )
            body = document.body[start:end].strip()
            if body:
                sections.append((heading, body))

    # Merge undersized sections forward so no chunk is a lone sentence.
    merged: list[tuple[str, str]] = []
    for heading, body in sections:
        if merged and len(body) < MIN_CHUNK_CHARS:
            prev_heading, prev_body = merged[-1]
            merged[-1] = (prev_heading, f"{prev_body}\n\n## {heading}\n\n{body}")
        else:
            merged.append((heading, body))

    chunks: list[DocumentChunk] = []
    for heading, body in merged:
        for part in _split_long_section(body, MAX_CHUNK_CHARS):
            # The title and heading are prepended into the chunk *text* so the
            # embedding sees them and so the fragment reads sensibly alone.
            text = f"{title} — {heading}\n\n{part}"
            digest = hashlib.blake2b(
                f"{document.path.name}:{heading}:{part[:120]}".encode(),
                digest_size=8,
            ).hexdigest()
            chunks.append(
                DocumentChunk(
                    id=f"{document.path.stem}-{digest}",
                    text=text,
                    title=title,
                    section=heading,
                    source=source,
                    category=category,
                    attribution=attribution,
                )
            )
    return chunks


def load_knowledge_base(directory: Path | None = None) -> list[DocumentChunk]:
    """Read and chunk every Markdown document in the knowledge directory."""
    base = directory or KNOWLEDGE_DIR
    if not base.exists():
        logger.warning("knowledge directory missing", extra={"path": str(base)})
        return []

    all_chunks: list[DocumentChunk] = []
    for path in sorted(base.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        frontmatter, body = parse_frontmatter(raw)
        document = LoadedDocument(path=path, frontmatter=frontmatter, body=body)
        chunks = chunk_document(document)
        document.chunks = chunks
        all_chunks.extend(chunks)
        logger.debug(
            "document chunked",
            extra={"file": path.name, "chunks": len(chunks)},
        )

    logger.info(
        "knowledge base loaded",
        extra={"documents": len(list(base.glob("*.md"))), "chunks": len(all_chunks)},
    )
    return all_chunks
