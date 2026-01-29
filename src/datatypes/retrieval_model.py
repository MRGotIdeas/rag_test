from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Retrieval:
    text: str
    source: str  
    chunk_id: str                
    embedding: np.ndarray     
    distance: float   