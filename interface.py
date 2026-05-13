import streamlit as st
import os
import json
import time
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback

load_dotenv()

# Estrutura de dados para validar o JSON
class ResultadoAnalise(BaseModel):
    type: str = Field(default="text")
    text: str = Field(description="Resposta formatada em markdown")
    source: str = Field(description="Nome do documento analisado")
    suggestions: list[str] = Field(description="3 sugestões de perguntas")

# Configuração da página
st.set_page_config(page_title="Analisador Active-BI", page_icon="📊", layout="wide")

# Estilo para o botão e áreas de métricas
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .metric-container { background-color: #ffffff; padding: 10px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Analisador de Relatórios (Active-BI)")

# Barra lateral para upload
with st.sidebar:
    st.header("⚙️ Configurações")
    st.write("Interface desenvolvida para o desafio técnico.")
    
    meu_arquivo = st.file_uploader("Suba o PDF aqui", type="pdf")
    
    if meu_arquivo:
        with open("temp.pdf", "wb") as f:
            f.write(meu_arquivo.getbuffer())
        st.success("PDF carregado!")

# entrada do usuário
pergunta = st.text_input("O que você deseja saber?", placeholder="Ex: Qual o resumo deste documento?")

if st.button("Analisar PDF"):
    if not meu_arquivo:
        st.error("Por favor, selecione um arquivo PDF.")
    elif not pergunta:
        st.warning("A pergunta não pode estar vazia.")
    else:
        try:
            with st.spinner("Processando..."):
                
                # leitura do documento
                loader = PyPDFLoader("temp.pdf")
                paginas = loader.load()
                texto_completo = "\n".join([p.page_content for p in paginas])
                
                # setup do modelo
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.2,
                    api_key=os.getenv("OPENAI_API_KEY")
                )
                
                llm_json = llm.with_structured_output(ResultadoAnalise)
                
                prompt_base = """Você é um analista de BI ajudando um cliente.
                Responda com base no contexto abaixo.
                
                Regras:
                1. Resposta em Markdown (use títulos ## e negrito).
                2. 'source' deve ser o nome do arquivo.
                3. 'suggestions' deve ter 3 perguntas relevantes.
                
                ARQUIVO: {nome_arquivo}
                CONTEÚDO: {contexto}
                PERGUNTA: {pergunta}
                """
                
                prompt = ChatPromptTemplate.from_template(prompt_base)
                chain = prompt | llm_json
                
                #capturando os custos
                with get_openai_callback() as monitor_custo:
                    resposta = chain.invoke({
                        "nome_arquivo": meu_arquivo.name,
                        "contexto": texto_completo,
                        "pergunta": pergunta
                    })

                #exibição dos Resultados
                col_esq, col_dir = st.columns([2, 1])
                
                with col_esq:
                    st.success("Análise finalizada!")
                    st.markdown("### 📝 Resposta")
                    
                    # Efeito de digitação fluida
                    caixa_texto = st.empty()
                    texto_acumulado = ""
                    for pedaco in resposta.text.split(" "):
                        texto_acumulado += pedaco + " "
                        caixa_texto.markdown(texto_acumulado)
                        time.sleep(0.03)
                    
                    st.markdown("---")
                    st.markdown("### 💡 Sugestões")
                    for sug in resposta.suggestions:
                        st.write(f"- {sug}")

                with col_dir:
                    # Custos
                    st.info("### 💰 Custos da Chamada")
                    m_col1, m_col2 = st.columns(2)
                    m_col1.metric("Tokens", monitor_custo.total_tokens)
                    m_col2.metric("Custo (USD)", f"${monitor_custo.total_cost:.4f}")
                    
                    st.markdown("---")
                    st.info("### 📦 JSON Gerado")
                    st.json(resposta.model_dump())
                    st.caption(f"Fonte: {resposta.source}")

        except Exception as e:
            st.error(f"Erro na análise: {str(e)}")
        finally:
            if os.path.exists("temp.pdf"):
                os.remove("temp.pdf")