from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import chromadb
from dotenv import load_dotenv
import os
import json
import re

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

def sanitize_json(raw: str) -> str:
    import re

    raw = raw.strip()
    raw = re.sub(r"^```json|```$", "", raw)  # Remove markdown code fences
    raw = re.sub(r",\s*([\]}])", r"\1", raw)  # Remove trailing commas
    raw = re.sub(r'\n', '', raw)  # Remove newlines

    # Fix unclosed string issue at the end
    if raw.count('{') > raw.count('}'):
        raw += '}'

    return raw

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
                "content": (
                    "You are a supportive and concise mental health coach. Based on the user's mood and journal input, "
                    "return your response strictly in the following JSON format:\n\n"
                    "{\n"
                    '  "coping_tips": ["tip 1", "tip 2"],\n'
                    '  "activity": "a helpful activity",\n'
                    '  "song": "a relaxing song"\n'
                    "}\n\n"
                    "Rules:\n"
                    "- Only provide JSON.\n"
                    "- Use correct JSON syntax with double quotes for all strings.\n"
                    "- Do NOT include explanations or comments.\n"
                    "- `coping_tips` must be a list of exactly TWO short, separate string tips.\n"
                    "- `song` must be a plain string: no parentheses, no extra notes.\n"
                )
            },
            {
                "role": "user",
                "content": f"""
                The user's current mood is: {data.mood}.
                Their journal entry says: "{data.journal_text}".
                Relevant context: {context}.

                Provide your advice using the exact format shared above.
                """
            }
        ]
    )

    raw_response = response.choices[0].message.content
    cleaned_response = sanitize_json(raw_response)

    try:
        parsed_advice = json.loads(cleaned_response)
        return {"advice": parsed_advice}
    except json.JSONDecodeError as e:
        return {
            "error": "Failed to parse model response",
            "raw": raw_response,
            "cleaned": cleaned_response,
            "details": str(e)
        }
