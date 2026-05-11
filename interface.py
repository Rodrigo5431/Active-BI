import streamlit as st
import os
import json
import time
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Carrega as chaves do .env
load_dotenv()

# Molde que vai forçar a IA a responder em JSON
class ResultadoAnalise(BaseModel):
    type: str = Field(default="text")
    text: str = Field(description="Texto formatado em markdown com titulos e negrito")
    source: str = Field(description="Nome do pdf que foi lido")
    suggestions: list[str] = Field(description="Lista com 3 sugestoes de perguntas")

# Configurando a página
st.set_page_config(page_title="Analisador Active-BI", page_icon="📊", layout="wide")

# Um css basico só pra deixar o botão principal azulzinho
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Analisador de Relatórios (Desafio)")

# menu lateral
with st.sidebar:
    st.header("⚙️ Configurações")
    st.write("Interface extra do desafio técnico.")
    
    # Aqui o usuario faz upload
    meu_arquivo = st.file_uploader("Joga o PDF aqui", type="pdf")
    
    if meu_arquivo:
        # Gambiarra necessária: salvar o arquivo temporariamente no disco 
        # pro PyPDFLoader do Langchain conseguir ler depois
        with open("temp.pdf", "wb") as f:
            f.write(meu_arquivo.getbuffer())
        st.success("Arquivo carregado!")

# Area principal
pergunta = st.text_input("O que você quer saber sobre esse pdf?", placeholder="Ex: Resume os pontos principais")

if st.button("Analisar PDF"):
    # Validações antes de chamar a api
    if not meu_arquivo:
        st.error("Esqueceu de subir o PDF ali no menu esquerdo!")
    elif not pergunta:
        st.warning("Digita uma pergunta antes de clicar em analisar.")
    else:
        try:
            with st.spinner("Lendo o documento e pensando..."):
                
                # lendo o conteudo do pdf
                loader = PyPDFLoader("temp.pdf")
                paginas = loader.load()
                texto_completo = "\n".join([p.page_content for p in paginas])
                
                # 2. subindo a IA
                llm = ChatOpenAI(
                    model="openai/gpt-4o-mini",
                    temperature=0.2,
                    max_tokens=1500,
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    base_url="https://openrouter.ai/api/v1"
                )
                
                # amarra a regra do JSON no modelo
                llm_forca_json = llm.with_structured_output(ResultadoAnalise)
                
                # montando o prompt
                texto_prompt = """Você é um analista ajudando um cliente.
                Lê o documento abaixo e responde a pergunta.
                
                Importante:
                - Formata sua resposta em Markdown.
                - Pega o nome certinho do documento pra usar no campo source.
                - Cria 3 perguntas relacionadas pro campo suggestions.
                
                Doc: {nome_arquivo}
                Conteúdo: {contexto}
                Pergunta: {pergunta}
                """
                
                prompt = ChatPromptTemplate.from_template(texto_prompt)
                chain = prompt | llm_forca_json
                
                # mandando pra IA
                resposta = chain.invoke({
                    "nome_arquivo": meu_arquivo.name,
                    "contexto": texto_completo,
                    "pergunta": pergunta
                })

                col_esquerda, col_direita = st.columns([2, 1])
                
                with col_esquerda:
                    st.success("Pronto!")
                    st.markdown("### 📝 Resposta")
                    
                    # Efeito maquina de escrever (letrinha por letrinha)
                    def efeito_digitacao(texto):
                        for pedaco in texto.split(" "):
                            yield pedaco + " "
                            time.sleep(0.04)

                    st.write_stream(efeito_digitacao(resposta.text))
                    
                    st.markdown("---")
                    st.markdown("### 💡 Perguntas Sugeridas")
                    for sug in resposta.suggestions:
                        st.write(f"- {sug}")

                with col_direita:
                    st.info("📦 JSON Final")
                    # mostrando o json bruto aqui
                    st.json(resposta.model_dump())
                    st.caption(f"Fonte: {resposta.source}")

        except Exception as e:
            st.error(f"Deu ruim na execução: {str(e)}")
        finally:
            if os.path.exists("temp.pdf"):
                os.remove("temp.pdf")