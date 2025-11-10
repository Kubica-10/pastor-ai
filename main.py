import os
import re # Para a busca de palavras
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel as LangChainBaseModel, Field

# --- Carregar a Bíblia na Memória ---
# Isso acontece UMA VEZ quando o servidor liga
BIBLIA_TEXT = ""
try:
    with open("biblia.txt", "r", encoding="utf-8") as f:
        BIBLIA_TEXT = f.read()
    print("Bíblia carregada com sucesso na memória.")
except FileNotFoundError:
    print("ERRO: biblia.txt não encontrado! O servidor não pode funcionar.")
    BIBLIA_TEXT = "Erro: Arquivo biblia.txt não encontrado."

# --- Inicialização da API FastAPI ---
app = FastAPI(
    title="Pastor_AI API",
    description="Um Agente de IA para sermões baseados na Bíblia."
)

# --- Modelos de Dados (Pydantic) ---
# O que o usuário envia para nós
class QueryInput(BaseModel):
    query: str = "O que a Bíblia diz sobre perdão?"

# O que a LLM deve nos devolver
class RespostaBiblica(LangChainBaseModel):
    query_analisada: str = Field(description="A pergunta ou tema original do usuário.")
    resposta_baseada_na_biblia: str = Field(description="A resposta (sermão, explicação) gerada estritamente a partir do contexto bíblico fornecido.")
    versiculos_encontrados: list[str] = Field(description="Uma lista de 3 a 5 versículos ou trechos que foram encontrados e usados como base.")
    oracao_sugestao: str = Field(description="Uma oração curta ou sugestão de reflexão baseada na resposta.")

# --- Função de Busca (O "RAG-lite") ---
def buscar_contexto_biblico(query: str, texto_completo: str):
    print(f"Buscando contexto para: '{query}'")
    
    # Pega as primeiras palavras da query como palavras-chave
    palavras_chave = re.findall(r'\b\w{4,}\b', query.lower()) # Pega palavras com 4+ letras
    
    if not palavras_chave:
        palavra_principal = query.split()[0].lower().strip("?,.") # Pega a primeira palavra se falhar
    else:
        palavra_principal = palavras_chave[0] # Usa a primeira palavra relevante
        
    print(f"Palavra-chave principal identificada: {palavra_principal}")
    
    contexto_encontrado = []
    # 're.IGNORECASE' faz a busca ignorar maiúsculas/minúsculas
    # Encontra até 7 versículos/linhas
    for linha in texto_completo.splitlines():
        if re.search(r'\b' + re.escape(palavra_principal) + r'\b', linha, re.IGNORECASE):
            contexto_encontrado.append(linha)
            if len(contexto_encontrado) >= 7:
                break
    
    if not contexto_encontrado:
        print("Nenhum contexto encontrado pela busca.")
        return "Nenhum versículo específico encontrado para esta palavra-chave. Por favor, use seu conhecimento geral da Bíblia para responder." # Fallback
        
    print(f"Encontrados {len(contexto_encontrado)} versículos.")
    return "\n".join(contexto_encontrado)

# --- O Endpoint da API (O que você vai vender) ---
@app.post("/gerar_conteudo/", response_model=RespostaBiblica)
async def gerar_conteudo_endpoint(input_data: QueryInput):
    
    query = input_data.query
    
    # 1. Recuperar (Buscar na biblia.txt)
    contexto = buscar_contexto_biblico(query, BIBLIA_TEXT)
    
    # 2. Configurar a LLM (Groq)
    # Pega a chave API do ambiente do Render
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERRO: GROQ_API_KEY não foi encontrada nas variáveis de ambiente.")
        return {"error": "Chave da LLM não configurada no servidor."}
        
    llm = ChatGroq(model="llama3-8b-8192", api_key=api_key)
    structured_llm = llm.with_structured_output(RespostaBiblica)
    
    # 3. Criar o Prompt (Geração)
    system_prompt = f"""
    Você é o "Pastor_AI", um assistente teológico especialista na Bíblia.
    Sua ÚNICA fonte de verdade é o CONTEXTO BÍBLICO fornecido abaixo.
    NÃO invente informações. NÃO use conhecimento externo.
    
    Sua missão é:
    1. Analisar a PERGUNTA do usuário: "{query}"
    2. Usar **ESTRITAMENTE** os versículos do CONTEXTO BÍBLICO para formular uma resposta (seja um sermão, uma explicação ou conselhos).
    3. Se o CONTEXTO BÍBLICO for "Nenhum versículo específico encontrado...", informe ao usuário que você usará o conhecimento bíblico geral para responder.
    4. Você DEVE seguir o formato de saída JSON.
    
    CONTEXTO BÍBLICO FORNECIDO:
    {contexto}
    """
    human_prompt = "PERGUNTA: {query}"
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt)])
    chain = prompt | structured_llm
    
    print("Invocando a LLM com contexto...")
    try:
        # 'ainvoke' é a versão assíncrona
        resultado = await chain.ainvoke({"query": query})
        return resultado
    except Exception as e:
        print(f"Erro na LLM: {e}")
        return {"error": "Ocorreu um erro ao gerar a resposta da IA."}

# Rota de "saúde" para o Render saber que estamos vivos
@app.get("/")
def health_check():
    return {"status": "Pastor_AI está no ar!"}
