from sentence_transformers import SentenceTransformer

from embedders.embedding_model import EmbeddingModel

# "Qwen/Qwen3-Embedding-0.6B"


class SentenceTransformerEmbeddings(EmbeddingModel):
    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        device: str | None = None,
        normalize: bool = True,
    ):
        """
        model_name: embedding model name
        device: 'cpu', 'cuda', or None (auto)
        normalize: normalize vectors for cosine similarity
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize

    def embed(self, texts: list[str]) -> list[list]:
        """
        Embed a list of texts into vectors.
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list:
        """
        Embed a single query (convenience method).
        """
        return self.embed([query])[0]
