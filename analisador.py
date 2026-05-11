import os
import sys
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback

# carrega o .env com a chave da api
load_dotenv()

# Classe pra forçar a saida em json do jeito que pediram no desafio
class RespostaFinal(BaseModel):
    type: str = Field(default="text")
    text: str = Field(description="A resposta em markdown com titulos, listas etc")
    source: str = Field(description="O nome do arquivo pdf lido")
    suggestions: list[str] = Field(description="3 perguntas de acompanhamento sobre o texto")

def ler_arquivo_pdf(caminho):
    #  [ensei em usar o fitz (pymupdf) para ler o texto pq achei mais rapido
    # nos meus testes antes, mas o PyPDFLoader resolveu bem pra esse caso.
    loader = PyPDFLoader(caminho)
    paginas = loader.load()
    
    texto = ""
    for pagina in paginas:
        # pegando o texto de cada pagina e quebrando linha
        texto += pagina.page_content + "\n"
        
    # pega só o nome do arquivo no final do caminho, ex: relatorio.pdf
    nome_arquivo = os.path.basename(caminho)
    
    return texto, nome_arquivo

if __name__ == "__main__":
    # Checa se passou o arquivo e a pergunta no terminal
    if len(sys.argv) != 3:
        erro = {"error": "Faltou argumento! Passa o caminho do arquivo e a pergunta entre aspas."}
        print(json.dumps(erro))
        sys.exit(1)

    arquivo_pdf = sys.argv[1]
    pergunta_usuario = sys.argv[2]

    if not os.path.exists(arquivo_pdf):
        print(json.dumps({"error": "Pdf não encontrado no caminho informado."}))
        sys.exit(1)

    try:
        conteudo_pdf, nome_do_pdf = ler_arquivo_pdf(arquivo_pdf)
        
        # print("debug: carregou", len(conteudo_pdf), "caracteres do pdf") 
        
        # configurando a IA
        llm = ChatOpenAI(
            model="openai/gpt-4o-mini", # Voltamos pro modelo super estável das suas aulas!
            temperature=0.2,
            max_tokens=1500, # Mantemos a trava pra OpenRouter aprovar seu saldo
            api_key=os.getenv("OPENROUTER_API_KEY"), 
            base_url="https://openrouter.ai/api/v1"  
        )
        
        # Aqui é o segredo pra garantir que a resposta não vaze texto fora do JSON
        llm_estruturado = llm.with_structured_output(RespostaFinal)

        # Prompt mais direto e simples
        meu_prompt = """Você é um assistente de análise de dados. Leia o documento e responda a pergunta.
        
        Importante:
        1. O campo 'text' tem que ser formatado em Markdown (use negrito, listas e títulos ##).
        2. O campo 'source' tem que ser EXATAMENTE o nome do documento.
        3. Crie 3 perguntas relevantes para o campo 'suggestions'.

        DOCUMENTO ({nome_do_pdf}):
        {texto_do_documento}

        PERGUNTA DO USUÁRIO: {pergunta_usuario}
        """
        
        prompt_template = ChatPromptTemplate.from_template(meu_prompt)
        chain = prompt_template | llm_estruturado
        
        # Bônus do desafio: calculando o custo da chamada
        with get_openai_callback() as custo:
            resposta_ia = chain.invoke({
                "nome_do_pdf": nome_do_pdf,
                "texto_do_documento": conteudo_pdf,
                "pergunta_usuario": pergunta_usuario
            })
            
            # jogo o print do custo no stderr pra não quebrar a formatação do json no stdout
            sys.stderr.write(f"\n---> Bônus (Custo) <--- \nTokens totais: {custo.total_tokens}\nCusto estimado: ${custo.total_cost:.6f}\n\n")

        # Retorna o json lindão na tela
        print(resposta_ia.model_dump_json(indent=2))

    except Exception as erro:
        print(json.dumps({"error": f"Deu erro na execução: {str(erro)}"}))
        sys.exit(1)