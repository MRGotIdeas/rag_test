import pytest

from config import OLLAMA_URL
from llm.llm import OllamaClient


def test_ollama_call():
    client = OllamaClient(model_name="llama3.2:1b", base_url=OLLAMA_URL)
    result = client.generate(prompt="Say hello")
    print("result: ", result)
    
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_stream():
    client = OllamaClient(model_name="llama3.2:1b", base_url=OLLAMA_URL)
    chunks = client.generate_stream(prompt="Say hello")
    result = list(chunks)
    print("result: ", result)
    assert len(result) > 1
