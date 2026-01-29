import pytest

from datatypes.chunk_model import Chunk
from datatypes.embedded_chunk_model import EmbeddedChunk
from embedders.sentence_transformer_embeddings import SentenceTransformerEmbeddings


@pytest.fixture
def chunks():
    chunks = [
        Chunk(text="Service du Premier ministre, le groupement interministériel de contrôle (GIC) est un service à compétence. ",
              source = "test.md",
              chunk_index=0,
              start= 0,
              end=106,
              chunk_method='test'),
        Chunk(
           text='nationale chargé de mettre en œuvre le cadre légal du renseignement. ',
              source = "test.md",
              chunk_index=1,
              start= 107,
              end=175,
              chunk_method='test'
        ),
        Chunk(
            text='Il centralise les demandes de techniques de renseignement émises par les services, met en œuvre celles qui nécessitent le concours des opérateurs de communications électroniques ou des fournisseurs de services de communication sur Internet et contribue à la centralisation du renseignement recueilli à proximité des objectifs. ',
              source = "test.md",
              chunk_index=2,
              start= 176,
              end=502,
              chunk_method='test'
              ),
              Chunk(
                  text='Totalement transformé en 2016, le GIC évolue en permanence pour ajuster son organisation aux besoins des services de sécurité et de renseignement et pour s’adapter aux transformations du monde des communications électroniques. ',
                  source = "test.md",
                chunk_index=3,
                start= 503,
                end=729,
                chunk_method='test'
              )
              ]
    return chunks

def test_embed(chunks):
    em = SentenceTransformerEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")
    embed_chunk = [EmbeddedChunk(chunk=c, 
                                  embedding=em.embed(c.text), 
                                  model_name="paraphrase-multilingual-mpnet-base-v2") 
                                  for c in chunks]

    print(embed_chunk[0])
    assert len(embed_chunk) == 4