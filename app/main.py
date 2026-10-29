from fastapi import FastAPI
from pydantic import BaseModel
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="Orion Prep API")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class AskRequest(BaseModel):
    pergunta: str

class AskResponse(BaseModel):
    resposta: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "user", "content": req.pergunta}
            ]
        )
        return AskResponse(resposta=completion.choices[0].message.content)
    except Exception as e:
        # Agora o erro vai aparecer no Swagger ao invés de 500 genérico
        return AskResponse(resposta=f"ERRO: {str(e)}")