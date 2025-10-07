import numpy as np
from backend.config import client, index, EMBED_MODEL

def embed_texts(texts):
    embeddings = []
    for i in range(0, len(texts), 50):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i+50])
        embeddings += [d.embedding for d in resp.data]
    return np.array(embeddings)

def upsert_to_pinecone(vectors):
    index.upsert(vectors=vectors)
