from chunkers.text_chunker import TextChunker
from chunkers.utils import _paragraphs_with_spans
from datatypes.chunk_model import Chunk


class ContextualChunker(TextChunker):
    def __init__(self, window: int = 2):
        self.window = window

    def chunk(self, text: str, source: str) -> list[Chunk]:
        paragraphs = _paragraphs_with_spans(text)
        chunks = []
        index = 0

        for i, (_, start, end) in enumerate(paragraphs):
            start = max(0, i - self.window)
            end = min(len(paragraphs), i + self.window + 1)
            chunks.append(
                Chunk(
                    text=text[paragraphs[start][1] : paragraphs[end - 1][2]],
                    source=source,
                    chunk_index=index,
                    start=paragraphs[start][1],
                    end=paragraphs[end - 1][2],
                    chunk_method=self.__class__.__name__,
                )
            )
            index += 1
        return chunks
