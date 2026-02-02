import pytest

from datatypes.retrieval_model import Retrieval
from rag.rag import Rag


@pytest.fixture
def rag_instance():
    return Rag(index_path='vector_database/faiss_index.index', 
               metadata_path='vector_database/metadata.pkl',  
               llm_name="llama3.2:1b", 
               embedding_name="Qwen/Qwen3-Embedding-0.6B",
               prompt_env="src/prompts",
               dimension_embedding_stored=3)


@pytest.fixture
def retrieval():
    return [
        Retrieval(
            text='la mairie est ouverte de 8h à 16h',
            source='brochure mairie',
            chunk_id="1",
            embedding=[1,1,0],
            distance=0.8
        ),
        Retrieval(
            text="Le maire s'appelle Mr. Kiwi",
            source='investiture',
            chunk_id="2",
            embedding=[1,0,0],
            distance=0.5
        )
        ]


def test_rag_prompt(rag_instance, retrieval):
    response = rag_instance.create_prompt_question(retrieval = retrieval,
                                                   question='Quels sont les horaires de la mairie ?')
   
    assert isinstance(response, str)
    assert response == """Vous êtes un assistant administratif compétent et bienveillant chargé de fournir des informations fiables. 
Vous vous appuyez uniquement sur le contexte fourni et répondez de manière concise et chaleureuse.
Si la réponse n'est pas présente, dites : "Je ne sais pas".


Context:
la mairie est ouverte de 8h à 16h (from brochure mairie, dist=0.800)

Le maire s'appelle Mr. Kiwi (from investiture, dist=0.500)


Question:
Quels sont les horaires de la mairie ?

Answer:"""


def test_rag(rag_instance, retrieval):
    result = rag_instance.answer_question(question = "Quels sont les horaires de la mairie ?",
                                          n_retrieval = 2
                                          )
    
    print(result)
    assert isinstance(result, str)