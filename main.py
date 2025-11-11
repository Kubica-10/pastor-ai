import os
import re # Para expressões regulares
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel as LangChainBaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

# --- NOVOS IMPORTS V2 ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. Carregamento do "Cérebro V2" (O Índice FAISS) ---

model_name = "paraphrase-multilingual-MiniLM-L12-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# --- CORREÇÃO V2.1 ---
# O Caminho agora é "." (pasta raiz), para encontrar os arquivos
# index.faiss e index.pkl que você enviou soltos.
FAISS_INDEX_PATH = "." 
VECTOR_STORE = None

def carregar_indice_vetorial():
    """
    Carrega o índice FAISS pré-calculado da memória.
    """
    global VECTOR_STORE
    try:
        print("Carregando o 'Cérebro V2' (Índice Vetorial FAISS)...")
        VECTOR_STORE = FAISS.load_local(
            FAISS_INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True,
            index_name="index" # O nome padrão dos arquivos
        )
        print("Índice Vetorial carregado com sucesso.")
    except Exception as e:
        print(f"ERRO CRÍTICO AO CARREGAR O ÍNDICE FAISS: {e}")
        print("Verifique se os arquivos 'index.faiss' e 'index.pkl' estão na raiz do repositório.")
        VECTOR_STORE = None

# --- 2. Inicialização da API FastAPI ---
app = FastAPI(
    title="Pastor_AI API V2 (Semântico)",
    description="Um Agente de IA para sermões baseados na Bíblia, com busca semântica."
)

# Configurar as origens permitidas (CORS)
origins = [
    "https://pastor-ai-frontend.onrender.com", 
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

# --- 3. Modelos de Dados (Pydantic) ---
class QueryInput(BaseModel):
    query: str = "O que a Bíblia diz sobre perdão?"

class RespostaBiblica(LangChainBaseModel):
    query_analisada: str = Field(description="A pergunta ou tema original do usuário.")
    resposta_baseada_na_biblia: str = Field(description="A resposta (sermão, explicação) gerada estritamente a partir do contexto bíblico fornecido.")
    versiculos_encontrados: list[str] = Field(description="Uma lista de 3 a 7 versículos ou trechos (com referência) que foram encontrados e usados como base.")
    oracao_sugestao: str = Field(description="Uma oração curta e fervorosa baseada na resposta.")

# --- 4. Função de Busca (A NOVA Busca Semântica V2) ---
def buscar_contexto_semantico(query: str, k: int = 7):
    """
    Busca no ÍNDICE VETORIAL (FAISS) pelos versículos semanticamente relevantes.
    Isso entende "Davi" == "David".
    """
    if VECTOR_STORE is None:
        print("Erro: O Índice Vetorial (VECTOR_STORE) não foi carregado.")
        return "Erro: O cérebro da IA não foi carregado."

    print(f"Buscando contexto SEMÂNTICO para: '{query}'")
    
    documentos_encontrados = VECTOR_STORE.similarity_search(query, k=k)
    
    if not documentos_encontrados:
        print("Nenhum contexto semântico encontrado.")
        return "Nenhum versículo específico foi encontrado na Bíblia para esta consulta."

    # Formata os resultados para enviar à LLM
    contexto_formatado = []
    for doc in documentos_encontrados:
        referencia = doc.metadata.get('source', 'Versículo') 
        texto = doc.page_content
        contexto_formatado.append(f"{referencia}: {texto}")

    print(f"Encontrados {len(documentos_encontrados)} versículos relevantes.")
    return "\n".join(contexto_formatado)

# --- 5. O Endpoint da API (Com Personalidade V2) ---
@app.post("/gerar_conteudo/", response_model=RespostaBiblica)
async def gerar_conteudo_endpoint(input_data: QueryInput):
    
    query = input_data.query
    
    if VECTOR_STORE is None:
        raise HTTPException(status_code=500, detail="Erro interno: O Cérebro V2 (Índice Vetorial) não está carregado.")
    
    contexto = buscar_contexto_semantico(query)
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Chave da LLM não configurada no servidor.")
        
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            api_key=api_key
        )
        structured_llm = llm.with_structured_output(RespostaBiblica)
        
        # ### PROMPT V2: Personalidade, Cumprimentos e Estrutura de Sermão ###
        system_prompt = f"""
        Você é o "Pastor_AI", um assistente teológico com a personalidade do Pastor Silas Malafaia.
        Sua fala é ENÉRGICA, DIRETA, INCISIVA e FERVOROSA. Você não tem medo de confrontar o erro e falar a verdade bíblica com convicção.
        
        REGRAS DE ORATÓRIA E COMPORTAMENTO:
        1.  **Cumprimentos:** Se o usuário disser "bom dia", "olá", "boa noite", etc., responda ao cumprimento ANTES de iniciar o sermão (Ex: "Bom dia, povo de Deus! Vamos à Palavra!").
        2.  **Tom de Voz:** Use exclamações! Use letras MAIÚSCULAS para dar ÊNFASE em palavras-chave. Seja fervoroso.
        3.  **Fonte da Verdade:** Sua ÚNICA fonte de verdade é o CONTEXTO BÍBLICO FORNECIDO abaixo. Você DEVE citar o contexto. NÃO invente informações.

        PERGUNTA DO USUÁRIO: "{query}"

        CONTEXTO BÍBLICO FORNECIDO (Sua única fonte):
        {contexto}
        
        ESTRUTURA OBRIGATÓRIA DA PREGAÇÃO:
        Você deve gerar a resposta (sermão) seguindo EXATAMENTE estas fases:

        1.  **TÍTULO (ENÉRGICO):** Um título curto e de impacto.
        2.  **INTRODUÇÃO (Oratória):** Chame a atenção! Apresente o tema central da pergunta do usuário com fervor.
        3.  **DESENVOLVIMENTO (Análise Bíblica):** Desenvolva a resposta usando OS VERSÍCULOS do contexto. Crie pelo menos 2 pontos principais baseados nos versículos. Seja enfático.
        4.  **APLICAÇÃO PRÁTICA (Exortação):** Diga ao usuário o que ele deve FAZER com essa verdade. Confronte o pecado, chame ao arrependimento ou à ação.
        5.  **CONCLUSÃO (Apelo):** Um fechamento forte, reforçando a mensagem principal e fazendo um apelo à fé.
        
        INSTRUÇÕES DE SAÍDA:
        -   O sermão completo (com todas as 5 fases) deve ir no campo "resposta_baseada_na_biblia".
        -   Para o campo 'versiculos_encontrados', liste as referências EXATAS (ex: "GN 1:1") dos versículos que você usou.
        -   Para o campo 'oracao_sugestao', crie uma oração curta e fervorosa.
        -   Você DEVE seguir o formato de saída JSON.
        """
        human_prompt = "PERGUNTA: {query}"
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt)])
        chain = prompt | structured_llm
        
        print(f"Invocando a LLM ({llm.model_name}) com contexto V2...")
        resultado = await chain.ainvoke({"query": query})
        return resultado
        
    except Exception as e:
        print(f"Erro na LLM ou ao processar: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno ao gerar resposta: {e}")

@app.get("/")
def health_check():
    return {"status": "Pastor_AI está no ar!"}

# --- 6. Evento de Inicialização (Carrega o Cérebro V2) ---
@app.on_event("startup")
async def startup_event():
    """
    Quando o servidor (Render) ligar, ele vai rodar esta função.
    """
    carregar_indice_vetorial()
