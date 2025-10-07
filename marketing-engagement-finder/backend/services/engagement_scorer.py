import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from backend.services.embeddings import embed_texts, upsert_to_pinecone
from backend.services.emotion_analysis import get_emotion_model, emotion_to_marketing

CONTEXT_CONFIGS = {
    "motivational": {
        "reference_phrases": ["overcoming challenges", "personal growth", "inspiration"],
        "emotion_weights": {"inspiration":0.5,"trust":0.3,"empathy":0.2},
        "weights": {"semantic":0.5,"emotion":0.5}
    },
    "promotional": {
        "reference_phrases": ["product benefits","value proposition"],
        "emotion_weights": {"authority":0.4,"trust":0.4,"curiosity":0.2},
        "weights": {"semantic":0.6,"emotion":0.4}
    },
    "social": {
        "reference_phrases": ["community","connection","relatable story"],
        "emotion_weights": {"empathy":0.4,"trust":0.4,"inspiration":0.2},
        "weights": {"semantic":0.5,"emotion":0.5}
    },
    "offer-based": {
        "reference_phrases": ["limited time offer","exclusive deal","discount"],
        "emotion_weights": {"urgency":0.6,"curiosity":0.4},
        "weights": {"semantic":0.7,"emotion":0.3}
    }
}

def score_engagement(transcript, context, top_n=5, pinecone_upsert=False):
    emo_model = get_emotion_model()
    cfg = CONTEXT_CONFIGS[context]
    texts = [t["text"] for t in transcript]

    emb = embed_texts(texts)
    ref_emb = embed_texts(cfg["reference_phrases"])
    ref_avg = np.mean(ref_emb, axis=0, keepdims=True)
    sem_scores = cosine_similarity(emb, ref_avg).flatten()

    emo_raw = emo_model(texts)
    emo_scores = []
    for e in emo_raw:
        mapped = emotion_to_marketing(e)
        weighted = sum(mapped.get(k, 0)*w for k, w in cfg["emotion_weights"].items())
        emo_scores.append(weighted)
    emo_scores = np.array(emo_scores)

    total = (cfg["weights"]["semantic"]*sem_scores) + (cfg["weights"]["emotion"]*emo_scores)

    if pinecone_upsert:
        vectors = [(str(i), emb[i].tolist(), {"text": texts[i], "score": float(total[i])}) for i in range(len(texts))]
        upsert_to_pinecone(vectors)

    res = [{
        "start": c["start"], "end": c["end"], "text": c["text"],
        "semantic_score": float(sem_scores[i]),
        "emotion_score": float(emo_scores[i]),
        "engagement_score": float(total[i])
    } for i, c in enumerate(transcript)]
    res.sort(key=lambda x: x["engagement_score"], reverse=True)
    return res[:top_n]
