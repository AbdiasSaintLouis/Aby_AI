from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import List
import logging
from prometheus_client import Counter, Summary, generate_latest, CONTENT_TYPE_LATEST

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change as needed for production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Download NLTK data
nltk.download("punkt_tab")
nltk.download("punkt")
nltk.download("stopwords")

# Workflow definitions
fluxos = {
    "X": ["Y1", "Y2", "Y3"],
}

OLLAMA_URL = "http://localhost:11434/api/generate"

# Prometheus metrics
PREDICTION_COUNT = Counter("inferences_total", "Número total de inferências")
PREDICTION_TIME = Summary("inference_duration_seconds", "Duração da inferência")

class Pergunta(BaseModel):
    pergunta: str

def extrair_palavras_chave(texto: str) -> List[str]:
    """Extrai palavras-chave da pergunta"""
    stop_words = set(stopwords.words("portuguese"))
    tokens = word_tokenize(texto.lower())  # Tokeniza e converte em minúsculas
    palavras_chave = [t for t in tokens if t.isalnum() and t not in stop_words]
    return palavras_chave

def determinar_fluxo(palavras_chave: List[str]) -> str:
    """Identifica o fluxo de trabalho baseado nas palavras-chave"""
    for fluxo, palavras in fluxos.items():
        # Verifica se alguma palavra-chave do fluxo aparece nas palavras extraídas
        if any(palavra in palavras_chave for palavra in palavras):
            return f"{fluxo.capitalize()}"
    return "Nenhum fluxo específico identificado."

def obter_resposta_llama(pergunta: str) -> str:
    """Obtém a resposta do modelo LLaMA"""
    try:
        payload = {
            "model": "tinyllama",
            "prompt": pergunta,
            "stream": False
        }
        resposta = requests.post(OLLAMA_URL, json=payload)
        resposta.raise_for_status()  # Raise exception for HTTP errors
        return resposta.json().get("response", "Erro ao obter resposta")
    except Exception as e:
        logging.error(f"Error fetching response from LLaMA: {e}")
        return "Erro ao obter resposta"

@app.get("/")
def home():
    return {"message": "API llama funcionando!"}

@app.get("/metrics")
def get_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/pergunta/")
async def fazer_pergunta(pergunta: Pergunta):
    """Processa a pergunta, envia ao LLaMA e retorna a resposta"""
    palavras_chave = extrair_palavras_chave(pergunta.pergunta)
    fluxo = determinar_fluxo(palavras_chave)
    
    # Obter resposta do LLaMA
    resposta_llama = obter_resposta_llama(pergunta.pergunta)
    
    return {"fluxo": fluxo, "resposta": resposta_llama}
