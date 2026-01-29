import pytest

from datatypes.chunk_model import Chunk
from chunkers.sliding_window_chunk import SlidingWindowChunker


@pytest.fixture
def text():
    text = """
    # L'Importance du Découpage de Texte (Chunking)

    Le découpage de texte, ou "chunking", est une technique fondamentale dans le traitement automatique du langage naturel (NLP) et la gestion de l'information. Elle consiste à diviser un long texte en morceaux plus petits et plus gérables, appelés "chunks". Cette étape est souvent cruciale avant d'appliquer d'autres traitements, comme l'indexation pour les moteurs de recherche, l'analyse de sentiments, ou l'alimentation de modèles de langage étendus (LLMs) dans des systèmes comme Retrieval-Augmented Generation (RAG).

    ## Pourquoi découper le texte ?

    Les raisons sont multiples. Premièrement, de nombreux modèles d'IA ont une limite sur la taille du contexte qu'ils peuvent traiter en une seule fois (la "context window"). Découper le texte permet de respecter ces limites. Deuxièmement, pour des tâches comme la recherche d'information, analyser des segments plus courts et ciblés peut donner de meilleurs résultats que d'analyser un document entier d'un seul bloc. Cela permet d'identifier plus précisément les passages pertinents. Enfin, cela facilite la parallélisation des traitements sur de grands volumes de données textuelles.

    ## Différentes Approches de Découpage

    Il n'existe pas une unique "bonne" façon de découper un texte. Le choix de la méthode dépend fortement du type de contenu et de l'objectif final. Voici quelques approches courantes que nous pouvons comparer :

    * **Le découpage récursif par caractères :** Cette méthode vise à créer des chunks d'une taille approximativement fixe (en nombre de caractères), avec un certain chevauchement pour ne pas perdre le contexte aux frontières. Elle essaie de couper sur des séparateurs logiques (paragraphes `\n\n`, phrases `. `, mots ` `) avant de couper brutalement si nécessaire pour respecter la taille. C'est **simple mais potentiellement abrupt**.

    * **Le découpage basé sur la structure Markdown :** Idéal pour les contenus rédigés en Markdown. Cette technique utilise la structure inhérente du document (titres `#`, `##`, etc., paragraphes séparés par des lignes vides, listes `*` ou `-`, blocs de code ``` ```, citations `>`) pour définir les chunks. Un parser Markdown peut identifier ces éléments, et chacun peut devenir un chunk. Cela **respecte la logique organisationnelle** du document voulue par l'auteur.

    * **Le découpage sémantique :** Cette approche plus avancée utilise des modèles NLP pour identifier les frontières naturelles du discours, comme la fin des phrases ou des groupes de phrases traitant d'un même sous-sujet. L'objectif est de créer des chunks qui sont *cohérents sémantiquement*, même si leur taille varie. Cela nécessite des outils plus complexes (comme spaCy ou des modèles d'embedding) mais peut préserver le sens de manière plus efficace.

    ## Exemple Concret et Comparaison

    Imaginons appliquer ces trois techniques à ce document Markdown même.
    Le découpage récursif pourrait couper le milieu d'un paragraphe s'il dépasse la taille cible, en essayant d'abord de couper entre les paragraphes (sur `\n\n`) ou les phrases.
    Le découpage basé sur la structure Markdown créerait probablement un chunk distinct pour chaque titre (`#`, `##`), chaque paragraphe (bloc de texte séparé par une ligne vide), et chaque élément de la liste (`*`).
    Le découpage sémantique essaierait de regrouper les phrases qui parlent d'un même concept (par exemple, l'explication du découpage récursif) pour former un chunk, même si cela couvre plusieurs lignes ou éléments Markdown.

    ## Conclusion

    En conclusion, le choix de la technique de chunking est une étape de conception importante. Il faut considérer la nature du texte (structuré en Markdown ou non, long ou court) et l'usage qui sera fait des chunks (recherche, résumé, alimentation d'un LLM). Une analyse comparative sur des exemples réels comme celui-ci est souvent nécessaire pour choisir la méthode la plus adaptée à vos besoins spécifiques.
    """
    return text


def test_cut(text):
    swc = SlidingWindowChunker(chunk_size=100, overlap=10)
    chunks = swc.chunk(text=text, source="test.md")
    print("chunk 0 : ", chunks[0])
    print("n chunks : ", len(chunks))
    assert len(chunks) > 0
    assert all(hasattr(c, "chunk_id") for c in chunks)
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.chunk_id
