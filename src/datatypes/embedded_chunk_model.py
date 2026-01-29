from dataclasses import dataclass, field
from typing import Any

import numpy as np

from datatypes.chunk_model import Chunk


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk: Chunk                  
    embedding: np.ndarray     
    model_name: str    
    extra: dict[str, Any] = field(default_factory=dict)