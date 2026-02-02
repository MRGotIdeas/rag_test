import numpy as np
from jinja2 import Environment, FileSystemLoader

from config import OLLAMA_URL
from datatypes.retrieval_model import Retrieval
from embedders.sentence_transformer_embeddings import SentenceTransformerEmbeddings
from llm.llm import DOWNLOADED_MODELS, OllamaClient
from vectordatabase.faiss_store import FaissVectorStore


class Rag:
    def __init__(self, 
                 index_path:str, 
                 metadata_path:str, 
                 llm_name:DOWNLOADED_MODELS, 
                 embedding_name:str,
                 dimension_embedding_stored :int = 1024,
                 prompt_env:str="src/prompts",
                 prompt_name:str="rag_prompt.jinja" 
                 ):
                 
        self.db = FaissVectorStore(dimension=dimension_embedding_stored)
        self.db.load(index_path=index_path, metadata_path=metadata_path)
    
        self.ollama = OllamaClient(llm_name, base_url=OLLAMA_URL)
        self.embedder = SentenceTransformerEmbeddings(embedding_name)
        self.env = Environment(loader=FileSystemLoader(prompt_env))
        self.template = self.env.get_template(prompt_name)


    def create_prompt_question(self, retrieval: list[Retrieval], question:str)-> str:
        if len(retrieval) > 1:
            context = "\n\n".join([f"{d.text} (from {d.source}, dist={d.distance:.3f})" for d in retrieval])
        else :
            context = f"{retrieval.text} (from {retrieval.source}, dist={retrieval.distance:.3f})"
        prompt_text = self.template.render(context=context, question=question)
        return prompt_text

    
    def answer_question(self, question:str, n_retrieval:int=3) -> str :
        print("Embedding conversion : Wait please")
        query_vector = np.asarray(self.embedder.embed(question), 
                                  dtype=np.float32)

        # find retrieval
        print("Retrieval selection : Wait please")
        retrieval = self.db.similarity_search(query_vector=query_vector, 
                                              top_k=n_retrieval)

        # Build context
        print("Prompt creation : Wait please")
        prompt_text = self.create_prompt_question(retrieval= retrieval, question = question)
        
        print("LLM response : almost done ! ")
        return self.ollama.generate(prompt_text)