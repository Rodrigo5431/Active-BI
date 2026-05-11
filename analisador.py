import os
import sys
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback

# Carregando as credenciais do ambiente
load_dotenv()

# Definição da estrutura de dados para garantir que o JSON de saída 
# siga rigorosamente o padrão exigido no desafio.
class AnaliseDocumento(BaseModel):
    type: str = Field(default="text")
    text: str = Field(description="Conteúdo da resposta formatado em Markdown")
    source: str = Field(description="Nome do arquivo PDF processado")
    suggestions: list[str] = Field(description="Lista com exatamente 3 perguntas de acompanhamento")

def extrair_texto_pdf(caminho_arquivo):
    """
    Realiza a leitura do PDF e extrai o conteúdo textual. 
    Optei pelo PyPDFLoader pela facilidade de integração com o ecossistema LangChain.
    """
    if not os.path.exists(caminho_arquivo):
        return None, "Arquivo não encontrado."

    try:
        loader = PyPDFLoader(caminho_arquivo)
        paginas = loader.load()
        
        # Consolida o texto de todas as páginas em uma única string
        conteudo_completo = "\n".join([p.page_content for p in paginas])
        nome_arquivo = os.path.basename(caminho_arquivo)
        
        return conteudo_completo, nome_arquivo
    except Exception as e:
        return None, str(e)

if __name__ == "__main__":
    # Validação básica de argumentos via linha de comando
    if len(sys.argv) != 3:
        mensagem_erro = {
            "error": "Argumentos inválidos! Use: python analisador.py 'caminho/do/arquivo.pdf' 'Sua pergunta'"
        }
        print(json.dumps(mensagem_erro))
        sys.exit(1)

    caminho_pdf = sys.argv[1]
    pergunta_usuario = sys.argv[2]

    # Processamento inicial do documento
    texto_doc, info_pdf = extrair_texto_pdf(caminho_pdf)

    if texto_doc is None:
        print(json.dumps({"error": f"Erro ao ler o PDF: {info_pdf}"}))
        sys.exit(1)

    try:
        # Configuração do modelo de linguagem. 
        llm = ChatOpenAI(
            model="openai/gpt-4o-mini",
            temperature=0.2,
            max_tokens=1500,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )

        # Forçamos o modelo a respeitar o esquema do Pydantic (Structured Output)
        llm_estruturado = llm.with_structured_output(AnaliseDocumento)

        # Definição do comportamento da IA através do prompt
        instrucao_prompt = """Você é um analista de dados especialista. 
        Sua tarefa é ler o documento fornecido e responder à pergunta do usuário.

        Regras de Resposta:
        1. O campo 'text' deve ser um Markdown limpo (use títulos ##, negrito e listas).
        2. O campo 'source' deve ser o nome exato do arquivo analisado.
        3. O campo 'suggestions' deve conter 3 perguntas relevantes para aprofundar a análise.

        DOCUMENTO: {nome_do_arquivo}
        CONTEÚDO: {conteudo_texto}

        PERGUNTA: {pergunta}
        """

        prompt_template = ChatPromptTemplate.from_template(instrucao_prompt)
        chain = prompt_template | llm_estruturado

        # Execução com monitoramento de custos (tokens e valores)
        with get_openai_callback() as monitor_custo:
            resultado = chain.invoke({
                "nome_do_arquivo": info_pdf,
                "conteudo_texto": texto_doc,
                "pergunta": pergunta_usuario
            })

            # Imprimimos os dados de custo no stderr para não "sujar" o stdout, 
            # mantendo o JSON de saída puro para o sistema que for consumir.
            sys.stderr.write(
                f"\n--- LOG DE EXECUÇÃO ---\n"
                f"Tokens Utilizados: {monitor_custo.total_tokens}\n"
                f"Custo Estimado: ${monitor_custo.total_cost:.6f}\n\n"
            )

        # Saída final em JSON formatado conforme o requisito obrigatório
        print(resultado.model_dump_json(indent=2))

    except Exception as erro_geral:
        print(json.dumps({"error": f"Erro interno na execução: {str(erro_geral)}"}))
        sys.exit(1)