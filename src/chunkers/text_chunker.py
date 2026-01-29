from abc import ABC, abstractmethod

from datatypes.chunk_model import Chunk


class TextChunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[Chunk]:
        pass
