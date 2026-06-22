"""
Split documents into overlapping text chunks for embedding and retrieval.
"""

def chunk_documents(documents, CHUNK_SIZE=1000, CHUNK_OVERLAP=200):
    """
    Split documents into overlapping chunks and preserve source metadata.

    Args:
        documents (list): Documents containing text and metadata.
        CHUNK_SIZE (int): Maximum characters per chunk.
        CHUNK_OVERLAP (int): Number of overlapping characters.

    Returns:
        list: Chunk dictionaries containing text, source, and page.
    """
    chunks = []

    for doc in documents:
        text = doc["text"]

        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE

            chunk = text[start:end]

            chunks.append({
                "text": chunk,
                "source": doc["source"],
                "page": doc["page"]
            })

            # Move forward while keeping overlap between chunks
            start += CHUNK_SIZE - CHUNK_OVERLAP

    print(f"Total chunks created: {len(chunks)}")
    print(chunks[0])

    return chunks