# Apprentissage RAG 

![alt text](image.png)


This project is for me to learn RAG. I created my own code of RAG following the really good free (🎅) [course on Openclassroom](https://openclassrooms.com/fr/courses/8532116-mettez-en-place-un-rag-pour-un-llm).

# Installation

This project uses data from the OpenClassrooms repository `8532116-mettez-en-place-un-rag-pour-un-llm`, included as a git submodule.

Clone the project with submodules:

```bash
git clone --recurse-submodules git@github.com:MRGotIdeas/rag_test.git
```

I work with a devcontainer. You'll find it in `.devcontainer/`

You'll need a Ollama client available and download some models. 
In the project, I used :
- `llama3.2:1b`, 
- `deepseek-r1:1.5b`,
- `qwen2.5:1.5b-instruct`,
- `qwen2.5:0.5b`.

If you use other models, you can add them in variable `DOWNLOADED_MODELS` in `src/llm/llm.py`. 
Create a `.env` file in your project with `OLLAMA_URL` set. 

# Use 
For converting all data files (pdf, audio, xlsx, etc.) into markdown, please do: 

```bash
python src/convert_docs_to_md.py
```

The converted documents into markdown are then in folder `processed_data`

For creating Faiss Vectordatabase index, please do:

```bash
python src/create_index.py
```

The Faiss index is then stored in folder `vector_database`.

To test Rag class, you can see `test/test_rag.py` but before, make sure you created the faiss index (#TODO create a fictive faiss vector database for tests).


To activate the web app, please do: 

```bash
streamlit run src/interface/app.py
```

