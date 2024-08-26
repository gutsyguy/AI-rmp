from dotenv import load_dotenv
import os
import google.generativeai as genai
import json
import openai
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Uncomment the below lines if you need to create the index
# pc.create_index(
#     name = "rag",
#     # Equal to number of embeddings
#     dimension= 768,
#     metric="cosine",
#     spec = ServerlessSpec("aws", "us-east-1")
# )

data = json.load(open("reviews.json"))

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel('gemini-1.5-flash')

processed_data = []
for review in data["reviews"]:
    response = genai.embed_content(
        model="models/text-embedding-004",
        content=review["reviews"],
    )

    # Extract the embedding list from the response
    embedding_values = response['embedding']  # Assuming the response has an 'embedding' key

    processed_data.append({
        "values": embedding_values,  # Use the list directly
        "id": review["professor"],
        "metadata": {
            "review": review["reviews"],
            "subject": review["subject"],
            "stars": review["stars"],
        }
    })

index = pc.Index("rag")
index.upsert(
    vectors=processed_data,
    namespace="ns1"
)

index.describe_index_stats()