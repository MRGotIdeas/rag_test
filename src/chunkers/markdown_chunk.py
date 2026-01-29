from datatypes.chunk_model import Chunk
from chunkers.text_chunker import TextChunker
from chunkers.utils import _markdown_with_spans


class MarkdownHeaderChunker(TextChunker):
    def chunk(self, text: str, source: str) -> list[Chunk]:
        sections = _markdown_with_spans(text)
        chunks = []
        for i, (section, start, end) in enumerate(sections):
            chunks.append(
                Chunk(
                    text=section.strip(),
                    source=source,
                    chunk_index=i,
                    start=start,
                    end=end,
                    chunk_method=self.__class__.__name__,
                )
            )

        return chunks
