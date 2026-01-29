import numpy as np
import pytest

from datatypes.chunk_model import Chunk
from datatypes.embedded_chunk_model import EmbeddedChunk
from vectordatabase.faiss_store import FaissVectorStore


def test_add_and_search():
    store = FaissVectorStore(dimension=3)

    chunks = [
        Chunk(
            text="hello world",
            source="doc1",
            chunk_index=0,
            start=0,
            end=4,
            chunk_method="test",
        ),
        Chunk(
            text="goodbye world",
            source="doc1",
            chunk_index=1,
            start=5,
            end=7,
            chunk_method="test",
        ),
    ]

    embedded_chunks = [
        EmbeddedChunk(chunk=chunks[0], model_name='test', embedding=np.array([1.0, 0.0, 0.0])),
        EmbeddedChunk(chunk=chunks[1], model_name='test', embedding=np.array([0.0, 1.0, 0.0])),
    ]

    # Act
    store.add(embedded_chunks)
    query = np.array([1.0, 0.0, 0.0])
    results = store.similarity_search(query, top_k=1)

    # Assert
    assert len(results) == 1
    assert results[0].text == "hello world"
