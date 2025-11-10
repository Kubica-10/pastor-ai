import os
import re # Para a busca de palavras
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel as LangChainBaseModel, Field

# --- Baixar e Carregar a Bíblia na Memória ---
def carregar_biblia():
    # CORREÇÃO FINAL: Este link está funcionando (Acabei de testar)
    biblia_url = "https://raw.githubusercontent.com/dvl/biblia/master/txt/acf.txt"
    try:
        print("Baixando a Bíblia (ACF) da internet...")
        response = requests.get(biblia_url)
        response.raise_for_status() # Verifica se o download deu erro
        print("Bíblia baixada com sucesso.")
        return response.text
    except Exception as e:
        print(f"ERRO CRÍTICO ao baixar a Bíblia: {e}")
        return "Erro: Não foi possível carregar o texto da Bíblia."

# Variável global que guarda o texto da Bíblia
BIBLIA_TEXT = carregar_biblia()
# -----------------------------------------------

# --- Inicialização da API FastAPI ---
app = FastAPI(
    title="Pastor_AI API",
    description="Um Agente de IA para sermões baseados na Bíblia."
)

# --- Modelos de Dados (Pydantic) ---
class QueryInput(BaseModel):
    query: str = "O que a Bíblia diz sobre perdão?"

class RespostaBiblica(LangChainBaseModel):
    query_analisada: str = Field(description="A pergunta ou tema original do usuário.")
    resposta_baseada_na_biblia: str = Field(description="A resposta (sermão, explicação) gerada estritamente a partir do contexto bíblico fornecido.")
    versiculos_encontrados: list[str] = Field(description="Uma lista de 3 a 5 versículos ou trechos que foram encontrados e usados como base.")
    oracao_sugestao: str = Field(description="Uma oração curta ou sugestão de reflexão baseada na resposta.")

# --- Função de Busca (O "RAG-lite") ---
def buscar_contexto_biblico(query: str, texto_completo: str):
    print(f"Buscando contexto para: '{query}'")
    palavras_chave = re.findall(r'\b\w{4,}\b', query.lower())
    
    if not palavras_chave:
        palavra_principal = query.split()[0].lower().strip("?,.")
    else:
        palavra_principal = palavras_chave[0]
        
    print(f"Palavra-chave principal identificada: {palavra_principal}")
    
    contexto_encontrado = []
    for linha in texto_completo.splitlines():
        if re.search(r'\b' + re.escape(palavra_principal) + r'\b', linha, re.IGNORECASE):
            # Limpa o versículo antes de adicionar
            versiculo_limpo = re.sub(r'^\S+\s\d+:\d+\s+', '', linha).strip()
            if versiculo_limpo: # Adiciona só se não estiver vazio
                contexto_encontrado.append(versiculo_limpo)
            if len(contexto_encontrado) >= 7:
                break
    
    if not contexto_encontrado:
        print("Nenhum contexto encontrado pela busca.")
        return "Nenhum versículo específico foi encontrado na Bíblia para esta palavra-chave. Por favor, use seu conhecimento bíblico geral para responder."
        
    print(f"Encontrados {len(contexto_encontrado)} versículos.")
    return "\n".join(contexto_encontrado)

# --- O Endpoint da API (O que você vai vender) ---
@app.post("/gerar_conteudo/", response_model=RespostaBiblica)
async def gerar_conteudo_endpoint(input_data: QueryInput):
    
    query = input_data.query
    contexto = buscar_contexto_biblico(query, BIBLIA_TEXT)
    
    if "Erro:" in contexto:
        raise HTTPException(status_code=500, detail="Erro interno: A Bíblia não pôde ser carregada.")
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERRO: GROQ_API_KEY não foi encontrada.")
        raise HTTPException(status_code=500, detail="Chave da LLM não configurada no servidor.")
        
    try:
        llm = ChatGroq(
            model="llama3-70b-8192", 
            api_key=api_key
        )
        structured_llm = llm.with_structured_output(RespostaBiblica)
        
        system_prompt = f"""
        Você é o "Pastor_AI", um assistente teológico especialista na Bíblia.
        Sua ÚNICA fonte de verdade é o CONTEXTO BÍBLICO fornecido abaixo.
        NÃO invente informações. NÃO use conhecimento externo.
        
        Sua missão é:
        1. Analisar a PERGUNTA do usuário: "{query}"
        2. Usar **ESTRITAMENTE** os versículos do CONTEXTO BÍBLICO para formular uma resposta (seja um sermão, uma explicação ou conselhos).
        3. Se o CONTEXTO BÍBLICO for "Nenhum versículo...", informe ao usuário que você usará o conhecimento bíblico geral para responder.
        4. Você DEVE seguir o formato de saída JSON.
        
        CONTEXTO BÍBLICO FORNECIDO:
        {contexto}
        """
        human_prompt = "PERGUNTA: {query}"
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt)])
        chain = prompt | structured_llm
        
        print("Invocando a LLM com contexto...")
        resultado = await chain.ainvoke({"query": query})
        return resultado
        
    except Exception as e:
        print(f"Erro na LLM ou ao processar: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno ao gerar resposta: {e}")

# Rota de "saúde" para o Render saber que estamos vivos
@app.get("/")
def health_check():
    return {"status": "Pastor_AI está no ar!"}
