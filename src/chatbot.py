import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Histórico - O bot precisa de memória
messages = [
    {"role": "system", "content": "Você é um engenheiro de IA sênior, direto ao ponto, que explica com código quando possível."}
]

print("Chatbot Orion Prep - Digite 'sair' para encerrar")
print("------------------------------------------------")

while True:
    user_input = input("\nVocê: ")
    if user_input.lower() == "sair":
        break

    messages.append({"role": "user", "content": user_input})

    print("\nIA: ", end="", flush=True)
    
    # Streaming - não trava a tela
    stream = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        temperature=0.7,
        stream=True
    )
    
    full_response = ""
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
            full_response += content
    
    print()  # nova linha
    messages.append({"role": "assistant", "content": full_response})