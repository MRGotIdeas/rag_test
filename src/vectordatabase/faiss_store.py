
import faiss
import numpy as np

from datatypes.embedded_chunk_model import EmbeddedChunk
from datatypes.retrieval_model import Retrieval


class FaissVectorStore:

    def __init__(self, dimension: int): 
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance
        self.index = faiss.IndexIDMap(self.index)  # allows custom integer IDs
        self.id_to_chunk: dict[int, EmbeddedChunk] = {}
        self.uuid_to_int: dict[str, int] = {}
        self.next_id = 0

    def add(self, embedded_chunks: list[EmbeddedChunk]):
        vectors = np.array([c.embedding for c in embedded_chunks], dtype=np.float32)
        ids = []
        for ec in embedded_chunks:
            # assign an integer ID for FAISS
            if ec.chunk.chunk_id not in self.uuid_to_int:
                int_id = self.next_id
                self.uuid_to_int[ec.chunk.chunk_id] = int_id
                self.next_id += 1
            else:
                int_id = self.uuid_to_int[ec.chunk.chunk_id]
            ids.append(int_id)
            self.id_to_chunk[int_id] = ec
        ids = np.array(ids, dtype=np.int64)
        self.index.add_with_ids(vectors, ids)

        
    def save(self, index_path: str, metadata_path: str):
        faiss.write_index(self.index, index_path)
        import pickle
        with open(metadata_path, "wb") as f:
            pickle.dump(self.id_to_chunk, f)


    def load(self, index_path: str, metadata_path: str):
        self.index = faiss.read_index(index_path)
        import pickle
        with open(metadata_path, "rb") as f:
            self.id_to_chunk = pickle.load(f)
        self.next_id = max(self.id_to_chunk.keys()) + 1



    def similarity_search(self, query_vector: list, top_k: int = 5) -> list[Retrieval]:
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = query_vector.astype(np.float32)

        distances, indices = self.index.search(query_vector, top_k)

        embedded_chunk_selected = [self.id_to_chunk[idx] for idx in indices[0]]
        distances = distances[0]

        best_retrieval = [
            Retrieval(
                text=s.chunk.text,
                source=s.chunk.source,
                chunk_id=s.chunk.chunk_id,
                embedding=s.embedding,
                distance = 1 / (1 + distances[i])
                ) 
                for i, s in enumerate(embedded_chunk_selected)
            ]
        
        return best_retrieval
    

    #  def similarity_search(self, query_vector: list, top_k: int = 5) -> list[Retrieval]:
    #     if query_vector.ndim == 1:
    #         query_vector = query_vector.reshape(1, -1)
    #         print("ici")
    #     query_vector = query_vector.astype(np.float32)

    #     distances, indices = self.index.search(query_vector, top_k)

    #     embedded_chunk_selected = [self.id_to_chunk[idx] for idx in indices[0]]

    #     best_retrieval = [
    #         Retrieval(
    #             text=s.chunk.text,
    #             source=s.chunk.source,
    #             chunk_id=s.chunk.chunk_id,
    #             embedding=s.embedding,
    #             similarity = 1 / (1 + distances[i])
    #             ) 
    #             for i, s in enumerate(embedded_chunk_selected)
    #         ]
        
    #     return best_retrieval

