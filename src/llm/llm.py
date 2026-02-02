import json
import os
from collections.abc import Generator
from typing import Literal

import requests

from config import OLLAMA_URL

DOWNLOADED_MODELS = Literal[
    "llama3.2:1b",
    "deepseek-r1:1.5b",
    "qwen2.5:1.5b-instruct",
]

class OllamaClient:
    def __init__(
        self,
        model_name: DOWNLOADED_MODELS = "llama3.2:1b",
        base_url: str | None = OLLAMA_URL,
    ):
        self.model_name = model_name
        self.base_url = base_url


    def generate_stream(
        self,
        prompt: str,
    ) -> Generator[str, None, None]:
        
        r = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": True,
            }
        )
        
        r.raise_for_status()
        
        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                yield data.get("response", "")


    def generate(
        self,
        prompt: str,
    ) -> str:
        
        r = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
            }
        )
        r.raise_for_status()
        return r.json()["response"]