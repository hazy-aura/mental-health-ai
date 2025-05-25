from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import chromadb
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure Groq as the OpenAI-compatible provider
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Initialize FastAPI app
app = FastAPI()

# Setup ChromaDB
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="coping_tips")

class MoodInput(BaseModel):
    mood: str
    journal_text: str = ""

@app.post("/generate-advice")
async def generate_advice(data: MoodInput):
    # Query ChromaDB
    results = collection.query(query_texts=[data.journal_text], n_results=2)
    context = "\n".join(results["documents"][0]) if results["documents"] else "None"

    # Generate response from LLaMA via Groq
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": "You are a supportive and concise mental health coach. Based on the user's mood and journal input, your job is to provide exactly two coping tips, one helpful activity, and one relaxing song. Do not include any additional advice, explanations, or commentary."
            },
            {
                "role": "user",
                "content": f"""
                The user's current mood is: {data.mood}.
                Their journal entry says: "{data.journal_text}".
                Relevant context: {context}.

                Based strictly on this information, suggest:
                1. Two specific coping tips (personalized to the mood and journal).
                2. One helpful activity (appropriate to their current emotional state).
                3. One relaxing song (that could help them unwind).

                Return only these four things—no extra text.
                """
            }
        ]
    )

    return {"advice": response.choices[0].message.content}
