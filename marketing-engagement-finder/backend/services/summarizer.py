import json
from backend.config import client

def summarize_highlights(top_chunks, context):
    joined = "\n\n".join(f"[{c['start']}–{c['end']}] {c['text']}" for c in top_chunks)
    sys = "You are a marketing highlight summarizer. Return JSON list {start,end,summary,why,ideal_clip_seconds}."
    user = f"Context: {context}\nSegments:\n{joined}"
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":sys},
            {"role":"user","content":user}
        ],
        response_format={"type":"json_object"}
    )
    return json.loads(resp.choices[0].message.content)
