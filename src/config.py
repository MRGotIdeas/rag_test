import os

from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL")

if not OLLAMA_URL:
    raise RuntimeError(
        "OLLAMA_URL is not set. "
        "Create a .env file or set the environment variable."
    )