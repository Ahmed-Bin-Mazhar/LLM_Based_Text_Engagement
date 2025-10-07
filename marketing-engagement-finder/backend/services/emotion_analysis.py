import numpy as np
from transformers import pipeline
from functools import lru_cache

MARKETING_EMOTIONS = {
    "inspiration": ["joy", "surprise"],
    "trust": ["neutral", "joy"],
    "urgency": ["surprise", "fear"],
    "curiosity": ["surprise", "anticipation"],
    "authority": ["confidence", "neutral"],
    "empathy": ["sadness", "joy"]
}

@lru_cache
def get_emotion_model():
    return pipeline("text-classification", 
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=True)

def emotion_to_marketing(emotion_output):
    base = {x["label"].lower(): x["score"] for x in emotion_output}
    mapped = {}
    for m_label, bases in MARKETING_EMOTIONS.items():
        mapped[m_label] = np.mean([base.get(b, 0.0) for b in bases])
    return mapped
