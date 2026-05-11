import streamlit as st
import os
import json
import time
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# puxa as chaves do arquivo .env
load_dotenv()

# Estrutura obrigatória para o JSON de saída 
class ResultadoAnalise(BaseModel):
    type: str = Field(default="text")
    text: str = Field(description="Resposta formatada em markdown")
    source: str = Field(description="Nome do documento analisado")
    suggestions: list[str] = Field(description="3 sugestões de perguntas")

# Configurações visuais da página
st.set_page_config(page_title="Analisador Active-BI", page_icon="📊", layout="wide")

# CSS rápido para deixar o botão principal com uma cara melhor
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Analisador de Relatórios (Desafio)")

# Barra lateral para configurações e upload
with st.sidebar:
    st.header("⚙️ Configurações")
    st.write("Interface bônus desenvolvida para o desafio.")
    
    # Input do arquivo PDF 
    meu_arquivo = st.file_uploader("Arraste seu PDF aqui", type="pdf")
    
    if meu_arquivo:
        # Salvando o arquivo temporariamente para o loader conseguir ler 
        with open("temp.pdf", "wb") as f:
            f.write(meu_arquivo.getbuffer())
        st.success("Arquivo carregado com sucesso!")

# Entrada da pergunta do usuário
pergunta = st.text_input("O que você deseja saber sobre este documento?", placeholder="Ex: Faça um resumo dos pontos principais")

if st.button("Analisar PDF"):
    if not meu_arquivo:
        st.error("Por favor, carregue um PDF antes de continuar.")
    elif not pergunta:
        st.warning("Você precisa digitar uma pergunta para a IA.")
    else:
        try:
            with st.spinner("Analisando o documento..."):
                
                # Extraindo o texto do PDF carregado 
                loader = PyPDFLoader("temp.pdf")
                paginas = loader.load()
                texto_completo = "\n".join([p.page_content for p in paginas])
                
                # Setup do modelo
                llm = ChatOpenAI(
                    model="openai/gpt-4o-mini",
                    temperature=0.2,
                    max_tokens=1500,
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    base_url="https://openrouter.ai/api/v1"
                )
                
                # Forçando a saída estruturada via Pydantic 
                llm_estruturado = llm.with_structured_output(ResultadoAnalise)
                
                # Prompt simulando o papel de analista de BI
                prompt_base = """Você é um analista de BI experiente auxiliando um cliente.
                Utilize o contexto abaixo para responder a pergunta de forma precisa.
                
                Regras obrigatórias:
                - Resposta sempre em Markdown (use títulos, listas e negrito).
                - Identifique o nome correto do arquivo no campo 'source'.
                - Gere exatamente 3 perguntas complementares no campo 'suggestions'.
                
                DOCUMENTO: {nome_arquivo}
                CONTEXTO: {contexto}
                PERGUNTA: {pergunta}
                """
                
                prompt = ChatPromptTemplate.from_template(prompt_base)
                chain = prompt | llm_estruturado
                
                # Chamada para o modelo
                resposta = chain.invoke({
                    "nome_arquivo": meu_arquivo.name,
                    "contexto": texto_completo,
                    "pergunta": pergunta
                })

                # Organização da exibição em colunas
                col_esq, col_dir = st.columns([2, 1])
                
                with col_esq:
                    st.success("Processamento finalizado!")
                    st.markdown("### 📝 Resposta do Analista")
                    
                    # Efeito de digitação (streaming visual)
                    placeholder = st.empty()
                    texto_acumulado = ""
                    for pedaco in resposta.text.split(" "):
                        texto_acumulado += pedaco + " "
                        placeholder.markdown(texto_acumulado)
                        time.sleep(0.04) # velocidade do efeito
                    
                    st.markdown("---")
                    st.markdown("### 💡 Próximos Passos")
                    for sug in resposta.suggestions:
                        st.write(f"- {sug}")

                with col_dir:
                    st.info("📦 JSON de Saída")
                    # Exibe o JSON
                    st.json(resposta.model_dump())
                    st.caption(f"Arquivo de origem: {resposta.source}")

        except Exception as e:
            st.error(f"Erro inesperado: {str(e)}")
        finally:
            # Limpeza do arquivo temporário
            if os.path.exists("temp.pdf"):
                os.remove("temp.pdf")