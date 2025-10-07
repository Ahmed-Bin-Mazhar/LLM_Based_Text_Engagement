import os
from openai import OpenAI
import pinecone

EMBED_MODEL = "text-embedding-3-large"
PINECONE_INDEX = "marketing-engagement"

# Initialize clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"))

# Pinecone index (create beforehand)
index = pinecone.Index(PINECONE_INDEX)
