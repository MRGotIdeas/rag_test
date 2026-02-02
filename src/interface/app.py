
import streamlit as st

from embedders.sentence_transformer_embeddings import SentenceTransformerEmbeddings
from rag.rag import Rag
import numpy as np


@st.cache_resource
def load_rag():
    return Rag(index_path="vector_database/faiss_index_minilm.index",
               metadata_path="vector_database/metadata_minilm.pkl",
               llm_name="qwen2.5:0.5b",
               embedding_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",#"Qwen/Qwen3-Embedding-0.6B",
               prompt_env="src/prompts",
               prompt_name="rag_prompt.jinja"
               )
    
rag = load_rag()

st.session_state.setdefault("history", [])

# ste = SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

st.set_page_config(page_title='Assistance de mairie', page_icon="🥳")

st.title("Assistant Virtuel de la Mairie")

question = st.chat_input("Comment puis-je vous aider?")
if question:
    with st.chat_message('user'):
        st.write(question)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.text("En train de réfléchir...")
        
        # vectors = np.asarray(ste.embed_query(question), dtype=np.float32)
        # retrieval = rag.db.similarity_search(query_vector=vectors, 
        #                                      top_k=3)
        # message_placeholder.write(retrieval)

        # message_placeholder.write(rag.create_prompt_question(retrieval=retrieval,
        #                                                      question=question))
        response = rag.answer_question(question)
        st.session_state.history.append((question, response))
        message_placeholder.write(response)

# placeholder = st.empty()
# text = ""

# for token in client.generate("Explain decorators in Python"):
#     text += token
#     placeholder.markdown(text)