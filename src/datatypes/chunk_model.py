import uuid
from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str

    # Metadata #
    source: str
    chunk_index: int
    start: int
    end: int
    chunk_method: str

    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    extra: dict[str, object] = field(default_factory=dict)
