# import spacy
import re

from datatypes.chunk_model import Chunk

from chunkers.text_chunker import TextChunker


class SentenceChunk(TextChunker):
    def __init__(self, max_chunk_size=300):
        self.max_chunk_size = max_chunk_size
        self._sentence_regex = re.compile(r"(?<=[.!?])\s+")

    def chunk(self, text: str, source: str) -> list[Chunk]:
        sentences = self._sentence_regex.split(text)

        chunks = []
        current_text = ""
        current_start = None
        chunk_index = 0

        cursor = 0  # tracks character position in original text

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_start = text.find(sentence, cursor)
            sentence_end = sentence_start + len(sentence)

            # If adding this sentence would exceed max size → flush
            if current_text and len(current_text) + len(sentence) > self.max_chunk_size:
                chunks.append(
                    Chunk(
                        text=current_text,
                        source=source,
                        chunk_index=chunk_index,
                        chunk_method="sentence",
                        start=current_start,
                        end=cursor,
                        extra={
                            "chunk_size": self.max_chunk_size,
                        },
                    )
                )
                chunk_index += 1
                current_text = ""
                current_start = None

            if current_text == "":
                current_start = sentence_start

            current_text += sentence + " "
            cursor = sentence_end

        # Flush last chunk
        if current_text:
            chunks.append(
                Chunk(
                    text=current_text.strip(),
                    source=source,
                    chunk_index=chunk_index,
                    chunk_method="sentence",
                    start=current_start,
                    end=cursor,
                    extra={
                        "sentence_count": len(self._sentence_regex.split(current_text)),
                    },
                )
            )

        return chunks

    # def chunk(self, text: str, source: str) -> List[Chunk] :
    #     doc = self.nlp(text)
    #     chunks = []
    #     current_segment = []

    #     index = 0
    #     for sent in doc.sents:
    #         if len(" ".join(current_segment)) + len(sent.text) <= self.max_chunk_size:
    #             current_segment.append(sent.text)
    #         else:
    #             if (" ".join(current_segment)).strip():
    #                 chunks.append(
    #                     Chunk(
    #                         text = " ".join(current_segment),
    #                         source = source,
    #                         chunk_index = index,
    #                         start = start,
    #                         end = end,
    #                         chunk_method = self.__class__.__name__,
    #                         extra = {'chunk_size':self.max_chunk_size}
    #                     )
    #                 )
    #                 index += 1
    #             current_segment = [sent.text]
    #     return chunks
