"""Deterministic word-based sliding-window chunking for documents."""


def chunk_document(
    text: str,
    doc_id: str,
    source_path: str,
    chunk_size: int = 800,
    overlap: int = 100
) -> list[dict]:
    """Split preprocessed text into overlapping word chunks.

    The function uses a sliding window over ``text.split()`` words.
    For each chunk:

    - ``start`` is the first word index in the chunk.
    - ``end`` is ``start + chunk_size`` (capped at total word count).
    - The next window starts at ``end - overlap``.

    This yields deterministic overlap while preventing a final redundant
    chunk that would be fully contained in the previous one.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    words: list[str] = text.split()
    if not words:
        return []

    chunks: list[dict] = []
    start: int = 0
    index: int = 0
    total_words: int = len(words)

    while start < total_words:
        end: int = min(start + chunk_size, total_words)
        chunk_words: list[str] = words[start:end]

        chunks.append(
            {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}chunk{index}",
                "text": " ".join(chunk_words),
                "source_path": source_path,
                "position": index,
            }
        )

        if end == total_words:
            break

        start = end - overlap
        index += 1

    return chunks
