import pytest

from embedders.sentence_transformer_embeddings import SentenceTransformerEmbeddings


@pytest.fixture
def text():
    return ["je fais un texte à embedder pour voir ce que ça va donner"]


def test_embeds(text):
    ste = SentenceTransformerEmbeddings()
    embeds = ste.embed(text)
    assert len(embeds[0]) == 768
    assert round(embeds[0][0], 5) == round(0.017945120111107826, 5)
