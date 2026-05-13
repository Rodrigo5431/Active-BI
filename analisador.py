import os
import sys
import json
from typing import Tuple, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback

load_dotenv()

# Definição do molde para o JSON de saída
class RespostaAnalise(BaseModel):
    type: str = Field(default="text")
    text: str = Field(description="Resposta formatada em Markdown com listas e títulos")
    source: str = Field(description="Nome do documento PDF")
    suggestions: list[str] = Field(description="3 perguntas de acompanhamento")

def carregar_dados_pdf(caminho: str) -> Tuple[Optional[str], str]:

    if not os.path.exists(caminho):
        return None, "Arquivo não encontrado."

    try:
        loader = PyPDFLoader(caminho)
        paginas = loader.load()
        
        texto_extraido = "\n".join([p.page_content for p in paginas]).strip()
        nome_doc = os.path.basename(caminho)
        
        # tratamento para PDFs
        if not texto_extraido:
            return None, "O PDF parece estar vazio ou é uma imagem escaneada sem texto (OCR necessário)."
            
        return texto_extraido, nome_doc
    except Exception as e:
        return None, str(e)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Uso correto: python analisador.py 'arquivo.pdf' 'pergunta'"}))
        sys.exit(1)

    caminho_arquivo = sys.argv[1]
    pergunta_usuario = sys.argv[2]

    conteudo_texto, nome_do_arquivo = carregar_dados_pdf(caminho_arquivo)

    if conteudo_texto is None:
        print(json.dumps({"error": f"Falha na leitura: {nome_do_arquivo}"}))
        sys.exit(1)

    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.2,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        llm_com_json = llm.with_structured_output(RespostaAnalise)

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """Você é um assistente técnico de BI extremamente analítico.
            Regras da Entrega:
            1. O campo 'text' deve OBRIGATORIAMENTE estar em Markdown (use títulos ## e listas).
            2. O campo 'source' deve ser o nome exato do arquivo enviado.
            3. O campo 'suggestions' deve conter exatamente 3 perguntas relacionadas para aprofundar a análise."""),
            
            ("user", """ARQUIVO: {nome_arquivo}
            CONTEÚDO DO DOCUMENTO: 
            {texto_pdf}

            PERGUNTA DO ANALISTA: {pergunta}""")
        ])

        workflow = prompt_template | llm_com_json

        with get_openai_callback() as calculo_custo:
            resultado_ia = workflow.invoke({
                "nome_arquivo": nome_do_arquivo,
                "texto_pdf": conteudo_texto,
                "pergunta": pergunta_usuario
            })

            sys.stderr.write(
                f"\n--- Métricas da Chamada ---\n"
                f"Tokens: {calculo_custo.total_tokens}\n"
                f"Custo Estimado: ${calculo_custo.total_cost:.6f}\n\n"
            )

        print(resultado_ia.model_dump_json(indent=2))

    except Exception as erro:
        print(json.dumps({"error": f"Ocorreu um erro na API: {str(erro)}"}))
        sys.exit(1)