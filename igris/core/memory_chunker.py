"""Memory Chunker: splits content into discrete chunks for the memory tree.

Part of GitHub issue #536: Memory Tree hierarchy.
"""
import hashlib
import re
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    """A single chunk of memory content."""
    chunk_id: str
    content: str
    offset: int  # character offset within source


class MemoryChunker:
    """Splits memory content into deterministic chunks of at most `max_tokens` tokens.

    Deterministic chunk ID = sha256(source_id + offset)[:16].
    Preserves paragraph and sentence boundaries where possible.
    """

    # Rough estimate: 1 token ≈ 4 characters
    CHARS_PER_TOKEN = 4

    def __init__(self, max_tokens: int = 3000):
        self.max_tokens = max_tokens
        self.max_chars = max_tokens * self.CHARS_PER_TOKEN

    def chunk(self, source_id: str, content: str) -> List[Chunk]:
        """Return a list of Chunk objects for the given source."""
        if not content:
            return []

        paragraphs = self._split_paragraphs(content)
        chunks = []
        current_chunks = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para)
            # If adding this paragraph would exceed max_chars, start a new chunk
            if current_len + para_len > self.max_chars and current_chunks:
                chunks.extend(self._finalize_chunks(source_id, current_chunks))
                current_chunks = []
                current_len = 0

            current_chunks.append(para)
            current_len += para_len

        if current_chunks:
            chunks.extend(self._finalize_chunks(source_id, current_chunks))

        return chunks

    def _split_paragraphs(self, content: str) -> List[str]:
        """Split content into paragraphs, preserving double newlines as separators."""
        # Split on one or more blank lines
        raw_paras = re.split(r'\n\s*\n', content)
        # Filter out empty paragraphs
        paras = [p.strip() for p in raw_paras if p.strip()]
        # Further split very long paragraphs by sentence boundaries
        result = []
        for para in paras:
            if len(para) > self.max_chars:
                result.extend(self._split_long_paragraph(para))
            else:
                result.append(para)
        return result

    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        """Split a long paragraph into sentence-level pieces, respecting max_chars."""
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        pieces = []
        current = []
        current_len = 0
        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > self.max_chars and current:
                pieces.append(' '.join(current))
                current = []
                current_len = 0
            current.append(sent)
            current_len += sent_len
        if current:
            pieces.append(' '.join(current))
        return pieces

    def _finalize_chunks(self, source_id: str, parts: List[str]) -> List[Chunk]:
        """Given a sequence of text parts, merge them into one or more chunks with deterministic IDs."""
        # Join parts with a single newline (preserve intra-chunk paragraph separation)
        text = '\n'.join(parts)
        offset = 0
        chunks = []
        while offset < len(text):
            # Take a slice up to max_chars but try to end at a sentence or word boundary
            end = min(offset + self.max_chars, len(text))
            # Backtrack to last sentence boundary if possible
            if end < len(text):
                # Look for last sentence-ending punctuation or space
                for i in range(end, offset, -1):
                    if text[i-1] in '.!?' or (i < len(text) and text[i-1] in '\n '):
                        end = i
                        break
                # If still too long, just cut at word boundary
                else:
                    for i in range(end, offset, -1):
                        if text[i-1] == ' ':
                            end = i
                            break
            chunk_text = text[offset:end].strip()
            chunk_id = self._make_chunk_id(source_id, offset)
            chunks.append(Chunk(chunk_id=chunk_id, content=chunk_text, offset=offset))
            offset = end
        return chunks

    def _make_chunk_id(self, source_id: str, offset: int) -> str:
        """Deterministic chunk ID: sha256(source_id + offset)[:16]."""
        raw = f"{source_id}::{offset}"
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]
