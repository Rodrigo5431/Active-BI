import os
import sys
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback

# Carrega as variáveis do arquivo .env
load_dotenv()

# Definição do molde para o JSON de saída conforme o PDF do desafio
class RespostaAnalise(BaseModel):
    type: str = Field(default="text")
    text: str = Field(description="Resposta formatada em Markdown com listas e títulos")
    source: str = Field(description="Nome do documento PDF")
    suggestions: list[str] = Field(description="3 perguntas de acompanhamento")

def carregar_dados_pdf(caminho):
    """
    Função simples para extrair o texto do PDF e pegar o nome do arquivo.
    """
    if not os.path.exists(caminho):
        return None, "Arquivo não encontrado."

    try:
        loader = PyPDFLoader(caminho)
        paginas = loader.load()
        
        # Junta o texto de todas as páginas
        texto_extraido = "\n".join([p.page_content for p in paginas])
        nome_doc = os.path.basename(caminho)
        
        return texto_extraido, nome_doc
    except Exception as e:
        return None, str(e)

if __name__ == "__main__":
    # Garante que o script recebeu o PDF e a Pergunta
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Uso correto: python analisador.py 'arquivo.pdf' 'pergunta'"}))
        sys.exit(1)

    caminho_arquivo = sys.argv[1]
    pergunta_usuario = sys.argv[2]

    # Inicia a extração
    conteudo_texto, nome_do_arquivo = carregar_dados_pdf(caminho_arquivo)

    if conteudo_texto is None:
        print(json.dumps({"error": f"Falha na leitura: {nome_do_arquivo}"}))
        sys.exit(1)

    try:
        # Configuração oficial da OpenAI
        llm = ChatOpenAI(
            model="gpt-4o-mini", # Modelo escolhido pela eficiência e suporte a JSON estruturado
            temperature=0.2,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Força o modelo a seguir o esquema da classe RespostaAnalise
        llm_com_json = llm.with_structured_output(RespostaAnalise)

        # Prompt focado em clareza e estrutura
        prompt_do_desafio = """Você é um assistente técnico de BI. 
        Analise o documento PDF abaixo e responda à pergunta do usuário de forma útil.

        Regras da Entrega:
        1. O campo 'text' deve estar em Markdown (use títulos ## e listas).
        2. O campo 'source' deve ser o nome do arquivo enviado.
        3. O campo 'suggestions' deve conter exatamente 3 perguntas relacionadas.

        ARQUIVO: {nome_arquivo}
        CONTEÚDO: {texto_pdf}

        PERGUNTA: {pergunta}
        """

        prompt_template = ChatPromptTemplate.from_template(prompt_do_desafio)
        workflow = prompt_template | llm_com_json

        # Monitoramento de tokens e custos para o bônus do desafio
        with get_openai_callback() as calculo_custo:
            resultado_ia = workflow.invoke({
                "nome_arquivo": nome_do_arquivo,
                "texto_pdf": conteudo_texto,
                "pergunta": pergunta_usuario
            })

            # Imprime os custos no stderr para manter o stdout com JSON limpo
            sys.stderr.write(
                f"\n--- Métricas da Chamada ---\n"
                f"Tokens: {calculo_custo.total_tokens}\n"
                f"Custo Estimado: ${calculo_custo.total_cost:.6f}\n\n"
            )

        # Exibe o JSON final formatado
        print(resultado_ia.model_dump_json(indent=2))

    except Exception as erro:
        print(json.dumps({"error": f"Ocorreu um erro na API: {str(erro)}"}))
        sys.exit(1)