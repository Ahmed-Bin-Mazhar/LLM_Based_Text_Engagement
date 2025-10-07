import json
import streamlit as st
import requests

API_URL = "http://localhost:8000/analyze"

st.set_page_config(page_title="Marketing Engagement Finder", layout="wide")
st.title("💡 Marketing Engagement Finder (API-based)")

uploaded = st.file_uploader("Upload transcript JSON (.txt/.json)", type=["txt", "json"])
top_n = st.slider("Top N highlights", 1, 10, 5)
override = st.selectbox("Override context", ["<auto>", "motivational", "promotional", "social", "offer-based"])
use_pinecone = st.checkbox("Upsert to Pinecone", value=False)

if uploaded:
    raw = uploaded.read().decode("utf-8")
    data = json.loads(raw)
    if isinstance(data, dict) and "transcript" in data:
        transcript = data["transcript"]
    else:
        transcript = data

    st.write(f"Total chunks: {len(transcript)}")

    if st.button("Analyze"):
        with st.spinner("Calling backend API..."):
            payload = {
                "transcript": transcript,
                "top_n": top_n,
                "override_context": None if override == "<auto>" else override,
                "use_pinecone": use_pinecone
            }
            res = requests.post(API_URL, json=payload)
            if res.status_code == 200:
                result = res.json()
                st.success(f"Context detected: {result['context_used']}")
                st.json(result["context_info"])
                st.subheader("Top Engaging Chunks")
                st.table([{
                    "start": c["start"],
                    "end": c["end"],
                    "engagement_score": round(c["engagement_score"], 4),
                    "text": c["text"][:150]+"…" if len(c["text"])>150 else c["text"]
                } for c in result["top_chunks"]])
                st.subheader("Summaries")
                st.json(result["summaries"])
                st.download_button("Download Results", json.dumps(result, indent=2), "engagement_results.json")
            else:
                st.error(f"Error {res.status_code}: {res.text}")
