from datatypes.chunk_model import Chunk
from chunkers.text_chunker import TextChunker


class SlidingWindowChunker(TextChunker):
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, source: str) -> list[Chunk]:
        chunks = []
        start = 0
        index = 0

        while start < len(text):
            end = start + self.chunk_size
            chunks.append(
                Chunk(
                    text=text[start:end],
                    source=source,
                    chunk_index=index,
                    start=start,
                    end=end,
                    chunk_method=self.__class__.__name__,
                    extra={
                        "chunk_size": self.chunk_size,
                        "overlap": self.overlap,
                    },
                )
            )

            start = end - self.overlap
            index += 1

        return chunks
