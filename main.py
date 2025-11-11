import os
import re # Para expressões regulares
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel as LangChainBaseModel, Field

# ### NOVO 1: Importar o CORSMiddleware ###
from fastapi.middleware.cors import CORSMiddleware


# --- 1. Carregamento da Bíblia (Lendo o arquivo local) ---
def carregar_biblia_local():
    """
    Lê o arquivo 'biblia.txt' local (que está no repositório) 
    para a memória.
    """
    arquivo_nome = "biblia.txt"
    try:
        print(f"Carregando a Bíblia (local) do arquivo: {arquivo_nome}...")
        with open(arquivo_nome, "r", encoding="utf-8") as f:
            texto = f.read()
        print("Bíblia local carregada com sucesso.")
        return texto
    except FileNotFoundError:
        print(f"ERRO CRÍTICO: O arquivo '{arquivo_nome}' não foi encontrado.")
        print("Certifique-se que 'biblia.txt' está no mesmo diretório que 'main.py' no GitHub.")
        return "Erro: Não foi possível carregar o texto da Bíblia."
    except Exception as e:
        print(f"ERRO CRÍTICO ao ler o arquivo da Bíblia: {e}")
        return "Erro: Não foi possível carregar o texto da Bíblia."

# --- 2. Otimização: Indexar a Bíblia (Processamento Único) ---
def processar_biblia(texto_completo: str):
    """
    Processa o texto puro da Bíblia e cria um "índice" em memória 
    (uma lista de tuplas) para busca rápida.
    """
    print("Processando e indexando a Bíblia (executado 1 vez)...")
    biblia_indexada = []
    referencia_regex = re.compile(r'^(\S+\s\d+:\d+)\s+(.*)')
    
    linhas_processadas = 0
    for linha in texto_completo.splitlines():
        match = referencia_regex.match(linha)
        if match:
            referencia = match.group(1)
            texto_versiculo = match.group(2).strip()
            if texto_versiculo:
                biblia_indexada.append((referencia, texto_versiculo))
                linhas_processadas += 1
    
    if linhas_processadas == 0:
        print("AVISO: Regex não encontrou referências (Ex: GN 1:1). Indexando por linha simples.")
        for i, linha in enumerate(texto_completo.splitlines()):
            texto_limpo = linha.strip()
            if texto_limpo and len(texto_limpo) > 20: 
                biblia_indexada.append((f"Linha {i+1}", texto_limpo))

    print(f"Bíblia indexada. Total de {len(biblia_indexada)} versículos/linhas.")
    return biblia_indexada

# --- Variáveis Globais (Carregadas na inicialização) ---
BIBLIA_TEXT_RAW = carregar_biblia_local()
BIBLIA_INDEXADA = []
if "Erro:" not in BIBLIA_TEXT_RAW:
    BIBLIA_INDEXADA = processar_biblia(BIBLIA_TEXT_RAW)

STOP_WORDS = set([
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "com", "não",
    "uma", "os", "na", "se", "nos", "como", "mas", "ao", "ele", "das", "à",
    "seu", "sua", "ou", "sobre", "qual", "foi", "ser", "por", "mais", "lhe",
    "diz", "bíblia"
])

# --- 3. Inicialização da API FastAPI ---
app = FastAPI(
    title="Pastor_AI API",
    description="Um Agente de IA para sermões baseados na Bíblia."
)

# ### NOVO 2: Configurar as origens permitidas (CORS) ###
# Lista dos sites que podem fazer requisições para sua API
origins = [
    "https://pastor-ai-frontend.onrender.com", # A URL do seu site
    "http://localhost", # Para testes locais no futuro
    "http://localhost:8080", # Para testes locais no futuro
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Permite as origens da lista
    allow_credentials=True,
    allow_methods=["*"], # Permite todos os métodos (GET, POST, etc)
    allow_headers=["*"], # Permite todos os cabeçalhos
)


# --- 4. Modelos de Dados (Pydantic) ---
class QueryInput(BaseModel):
    query: str = "O que a Bíblia diz sobre perdão?"

class RespostaBiblica(LangChainBaseModel):
    query_analisada: str = Field(description="A pergunta ou tema original do usuário.")
    resposta_baseada_na_biblia: str = Field(description="A resposta (sermão, explicação) gerada estritamente a partir do contexto bíblico fornecido.")
    versiculos_encontrados: list[str] = Field(description="Uma lista de 3 a 5 versículos ou trechos (com referência, se disponível) que foram encontrados e usados como base.")
    oracao_sugestao: str = Field(description="Uma oração curta ou sugestão de reflexão baseada na resposta.")

# --- 5. Função de Busca (O "RAG-lite" Otimizado) ---
def buscar_contexto_biblico(query: str, biblia_processada: list):
    print(f"Buscando contexto para: '{query}'")
    
    palavras_query = re.findall(r'\b\w+\b', query.lower())
    palavras_chave = [p for p in palavras_query if p not in STOP_WORDS and len(p) >= 2] 

    if not palavras_chave:
        palavras_chave = palavras_query[:1] 

    print(f"Palavras-chave identificadas: {palavras_chave}")

    contexto_encontrado = []
    textos_adicionados = set() 

    # Passa 1: Tenta encontrar versículos que contenham TODAS as palavras-chave
    for ref, texto_versiculo in biblia_processada:
        texto_lower = texto_versiculo.lower()
        if all(re.search(r'\b' + re.escape(chave) + r'\b', texto_lower) for chave in palavras_chave):
            if texto_versiculo not in textos_adicionados:
                contexto_encontrado.append(f"{ref}: {texto_versiculo}")
                textos_adicionados.add(texto_versiculo)

    # Passa 2: Se não achar nada, tenta com QUALQUER palavra-chave
    if not contexto_encontrado:
        print("Busca por 'TODAS' palavras falhou. Tentando 'QUALQUER' palavra...")
        for ref, texto_versiculo in biblia_processada:
            texto_lower = texto_versiculo.lower()
            if any(re.search(r'\b' + re.escape(chave) + r'\b', texto_lower) for chave in palavras_chave):
                 if texto_versiculo not in textos_adicionados:
                    contexto_encontrado.append(f"{ref}: {texto_versiculo}")
                    textos_adicionados.add(texto_versiculo)
                    if len(contexto_encontrado) >= 10: 
                        break 

    if not contexto_encontrado:
        print("Nenhum contexto encontrado pela busca.")
        return "Nenhum versículo específico foi encontrado na Bíblia para esta consulta. Por favor, use seu conhecimento bíblico geral para responder."

    contexto_final = "\n".join(contexto_encontrado[:7])
    print(f"Encontrados {len(contexto_encontrado)} versículos. Enviando 7 para a LLM.")
    return contexto_final

# --- 6. O Endpoint da API (O que você vai vender) ---
@app.post("/gerar_conteudo/", response_model=RespostaBiblica)
async def gerar_conteudo_endpoint(input_data: QueryInput):
    
    query = input_data.query
    
    if not BIBLIA_INDEXADA:
        raise HTTPException(status_code=500, detail="Erro interno: A Bíblia não pôde ser carregada.")
    
    contexto = buscar_contexto_biblico(query, BIBLIA_INDEXADA)
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Chave da LLM não configurada no servidor.")
        
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            api_key=api_key
        )
        structured_llm = llm.with_structured_output(RespostaBiblica)
        
        system_prompt = f"""
        Você é o "Pastor_AI", um assistente teológico especialista na Bíblia.
        Sua ÚNICA fonte de verdade é o CONTEXTO BÍBLICO FORNECIDO abaixo.
        NÃO invente informações. NÃO use conhecimento externo.
        
        Sua missão é:
        1. Analisar a PERGUNTA do usuário: "{query}"
        2. Usar **ESTRITAMENTE** os versículos do CONTEXTO BÍBLICO para formular uma resposta.
        3. Para o campo 'versiculos_encontrados', liste as referências EXATAS (ex: "GN 1:1" ou "Linha 123") dos versículos que você usou.
        4. Você DEVE seguir o formato de saída JSON.
        
        CONTEXTO BÍBLICO FORNECIDO (Sua única fonte):
        {contexto}
        """
        human_prompt = "PERGUNTA: {query}"
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt)])
        chain = prompt | structured_llm
        
        print(f"Invocando a LLM ({llm.model_name}) com contexto...")
        resultado = await chain.ainvoke({"query": query})
        return resultado
        
    except Exception as e:
        print(f"Erro na LLM ou ao processar: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno ao gerar resposta: {e}")

# ### NOVO 3: Adicionar uma rota de 'health check' para o CORS (opcional) ###
@app.get("/")
def health_check():
    return {"status": "Pastor_AI está no ar!"}
