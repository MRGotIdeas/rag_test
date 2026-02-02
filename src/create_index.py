from pathlib import Path

from tqdm import tqdm

from chunkers.sentence_chunk import SentenceChunk
from datatypes.embedded_chunk_model import EmbeddedChunk
from embedders.sentence_transformer_embeddings import SentenceTransformerEmbeddings
from vectordatabase.faiss_store import FaissVectorStore

BASE_DIR = Path(__file__).resolve().parent.parent
INPUTS_DIR = BASE_DIR / "processed_data"
OUTPUT_DIR = BASE_DIR / "vector_database"

index_path = OUTPUT_DIR / "faiss_index.index"
meta_path = OUTPUT_DIR / "metadata.pkl"

index_path.parent.mkdir(parents=True, exist_ok=True)
meta_path.parent.mkdir(parents=True, exist_ok=True)

# instanciate chunker - In the project, we chose sentenceChunk for simplicity
sc = SentenceChunk()

# instanciate embedding - In the project, we chose Qwen embedding
model_name="Qwen/Qwen3-Embedding-0.6B"
em = SentenceTransformerEmbeddings(model_name=model_name)

# instanciate faiss vector 
vector_store = None


for i, file_path in tqdm(enumerate(INPUTS_DIR.rglob("*"))):
    if not file_path.is_file():
        continue

    print("filepath: ", file_path)

    # read file
    with open(file_path, encoding="utf-8") as f:
        markdown_string = f.read()

    # chunk texts
    chunks = sc.chunk(text=markdown_string, source=str(file_path))

    if not chunks:
        continue

    # embed texts
    embedded_chunks = [EmbeddedChunk(chunk=c, 
                                     embedding=em.embed(c.text), 
                                     model_name=model_name) for c in chunks]
    
    # store in vector database 
    if vector_store is None:
        vector_store = FaissVectorStore(
            dimension=len(embedded_chunks[0].embedding)
        )
        if index_path.exists() and meta_path.exists():
            vector_store.load(str(index_path), str(meta_path))
    
    vector_store.add(embedded_chunks)

    if vector_store is not None:
        vector_store.save(str(index_path), str(meta_path))

# save vectordatabase every processed document
if vector_store is not None:
    vector_store.save(str(index_path), str(meta_path))
