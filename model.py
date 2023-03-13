import threading
from typing import Any, List

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import GPT2TokenizerFast

_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
_model.max_seq_length = 512
_model_lock = threading.Lock()


def encode(texts: List[str], batch_size: int = 32) -> np.ndarray:
    # While pytorch in inference mode should be thread safe, SentenceTransformers
    # is not. So we need to lock the model.
    with _model_lock:
        result = _model.encode(texts, batch_size=batch_size)

    return result


_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
_tokenizer.model_max_length = 2000


def tokenize(text: str | List[str], **kwargs) -> Any:
    result = _tokenizer(text, **kwargs)
    return result
