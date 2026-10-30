from fastapi import FastAPI
from pydantic import BaseModel
import os
from groq import Groq
from dotenv import load_dotenv

chunks_flamengo = load_txt("data/flamengo.txt")
collection, embed_model = build_db(chunks_flamengo)

load_dotenv() # 1
app = FastAPI(title="Orion Prep API") # 2
client = Groq(api_key=os.getenv("GROQ_API_KEY")) # 3

class AskRequest(BaseModel): # 4
    pergunta: str

class AskResponse(BaseModel):
    resposta: str

@app.get("/health") # 5
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse) # 6
def ask(req: AskRequest):
    try: # 7
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b", # 8
            messages=[
                {"role": "user", "content": req.pergunta}
            ]
        )
        return AskResponse(resposta=completion.choices[0].message.content)
    
    except Exception as e: # 9
        return AskResponse(resposta=f"ERRO: {str(e)}")