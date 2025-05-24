from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class MoodInput(BaseModel):
    mood: str
    journal_text: str = ""

@app.post("/generate-advice")
async def generate_advice(data: MoodInput):
    # Example static response for now
    return {
        "tips": [
            f"Try deep breathing exercises to help with your mood: {data.mood}"
        ],
        "songs": [
            "Weightless by Marconi Union"
        ]
    }
