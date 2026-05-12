# Analisador de Documentos com IA - Active BI

Ferramenta desenvolvida em Python que permite analisar relatórios em PDF através de perguntas em linguagem natural. O projeto entrega a funcionalidade principal via linha de comando (retornando um JSON rigorosamente estruturado) e também conta com uma interface gráfica web desenvolvida como bônus.

## 🚀 Como Executar

1. Clone este repositório:
   `git clone https://github.com/SeuUsuario/Active-BI.git`
2. Crie um ambiente virtual e ative-o:
   `python -m venv venv`
   * **Windows:** `venv\Scripts\activate`
   * **Linux/Mac:** `source venv/bin/activate`
3. Instale as dependências necessárias:
   `pip install -r requirements.txt`
4. Crie um arquivo `.env` na raiz do projeto e adicione a sua chave da API da OpenAI:
   `OPENAI_API_KEY=sk-suachaveaqui`

### 💻 Modo CLI (Requisito Principal do Desafio)
Execute o script passando o caminho do PDF e a pergunta (entre aspas). A saída será estritamente um JSON válido:
`python analisador.py "caminho/do/relatorio.pdf" "Quais foram os principais produtos vendidos?"`

### 🖥️ Modo Interface Gráfica (Bônus de UI/UX)
Para uma experiência amigável, com upload de arquivo arrastar-e-soltar e resposta gerada em tempo real (efeito streaming), execute:
`python -m streamlit run interface.py`

## 🧠 Justificativa da Escolha do Modelo

Para este desafio, optei pelo modelo **`gpt-4o-mini`** pelas seguintes razões:
1. **Suporte a Structured Outputs:** O modelo lida excepcionalmente bem com retornos forçados em esquemas JSON complexos validados pelo Pydantic. Isso é crucial para garantir que o script de linha de comando nunca "vaze" texto descritivo fora do objeto JSON.
2. **Custo-benefício (Contexto Longo):** Como a ferramenta lê PDFs (consumindo um volume alto de tokens de entrada), o `gpt-4o-mini` entrega um raciocínio lógico muito próximo aos modelos maiores, porém com um custo financeiro drasticamente menor, tornando a solução viável para o dia a dia do BI.
3. **Velocidade de Inferência:** Possui latência muito baixa, garantindo respostas rápidas no terminal e permitindo a implementação fluida do gerador de caracteres (efeito máquina de escrever) na interface gráfica.

## 💰 Estimativa de Custos (Bônus)

No modo CLI (`analisador.py`), o script utiliza o módulo `get_openai_callback` da LangChain para calcular os tokens consumidos (entrada + saída) e o custo estimado da chamada. 
Para não quebrar o requisito principal da vaga (que exige a saída OBRIGATÓRIA em JSON puro), os dados de estimativa de custo são impressos no canal de erro padrão (`sys.stderr`), separando visualmente o log do resultado estruturado.