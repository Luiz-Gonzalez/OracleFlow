import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from loaders import *
import tempfile
from langchain.prompts import ChatPromptTemplate


TIPOS_ARQUIVOS_VALIDOS = [
    'Website Link', 'YouTube Link', 'Pdf', 'Csv', 'Txt'
]

CONFIG_MODELOS = {
                'Groq' : 
                        {'modelos' : ['gemma2-9b-it', 'llama-3.1-8b-instant', 'llama-3.3-70b-versatile'],
                         'chat' : ChatGroq},
                'OpenAi' : 
                        {'modelos' : ['gpt-4.1-nano', 'gpt-4.1-mini', 'gpt-4o', 'o1-mini'],
                         'chat' : ChatOpenAI},
                'Gemini' : {'modelos' : ['gemini-2.5-pro', 'gemini-2.5-flash-lite'],
                            'chat' : ChatGoogleGenerativeAI}
                }

MEMORIA = ConversationBufferMemory()

def carrega_arquivos(tipo_arquivo, arquivo):
    if tipo_arquivo == 'Website Link':
        documento = carrega_site(arquivo)
    # youtube nao esta funcionando
    if tipo_arquivo == 'YouTube Link':
        documento = carrega_youtube(arquivo)
    if tipo_arquivo == 'Pdf':
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
            temp.write(arquivo.read()) # read é para ler o arquivo
            nome_temp = temp.name #vai ser o caminho do arquivo na pasta temp
        documento = carrega_pdf(nome_temp)
    if tipo_arquivo == 'Csv':
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_csv(nome_temp)
    if tipo_arquivo == 'Txt':
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
            temp.write(arquivo.read()) # read é para ler o arquivo
            nome_temp = temp.name #vai ser o caminho do arquivo na pasta temp
        documento = carrega_txt(nome_temp)
    return documento

def carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo):
 
    documento = carrega_arquivos(tipo_arquivo, arquivo)

    system_message = '''Você é um assistente amigável chamado Oráculo.
    Você possui acesso às seguintes informações vindas 
    de um documento {}: 

    ####
    {}
    ####

    Utilize as informações fornecidas para basear as suas respostas.

    Sempre que houver $ na sua saída, substita por S.

    Se a informação do documento for algo como "Just a moment...Enable JavaScript and cookies to continue" 
    sugira ao usuário carregar novamente o Oráculo!'''.format(tipo_arquivo, documento)
    
    template = ChatPromptTemplate.from_messages([
        ('system', system_message),
        ('placeholder', '{chat_history}'),
        ('user', '{input}')
    ])

    chat = CONFIG_MODELOS[provedor]['chat'](model=modelo, api_key=api_key)
    chain = template | chat

    st.session_state['chain'] = chain

def pagina_chat():
    st.header('🤖 Welcome to OracleFlow', divider=True)
    
    chain = st.session_state.get('chain')

    if chain is None:
        st.error('Fill all the informations on the Sidebar (document and model) and than Click on Load OracleFlow Button')
        st.stop()

    #pegar mensagens na memoria da sessão
    memoria = st.session_state.get('memoria', MEMORIA)
    
    for mensagem in memoria.buffer_as_messages:
        chat = st.chat_message(mensagem.type) # type quem enviou
        chat.markdown(mensagem.content) # conteudo da mensagem
    
    input_usuario = st.chat_input('Just tell me...')
    if input_usuario:
        chat = st.chat_message('human')
        chat.markdown(input_usuario)

        chat = st.chat_message('ai')
        resposta = chat.write_stream(chain.stream({
            'input' : input_usuario,
            'chat_history' : memoria.buffer_as_messages
            }))

        memoria.chat_memory.add_user_message(input_usuario)
        memoria.chat_memory.add_ai_message(resposta)
        st.session_state['memoria'] = memoria

def sidebar():
    tabs = st.tabs(['Documents upload', 'Model Selection'])
    with tabs[0]:
        tipo_arquivo = st.selectbox('Select', TIPOS_ARQUIVOS_VALIDOS)
        if tipo_arquivo == 'Website Link':
            arquivo = st.text_input('Type website URL')
        if tipo_arquivo == 'YouTube Link':
            arquivo = st.text_input('Type YouTube URL')
        if tipo_arquivo == 'Pdf':
            arquivo = st.file_uploader('Upload the Pdf file', type=['.pdf'])
        if tipo_arquivo == 'Csv':
            arquivo = st.file_uploader('Upload the Csv file', type=['.csv'])
        if tipo_arquivo == 'Txt':
            arquivo = st.file_uploader('Upload the Txt file', type=['.txt'])
    with tabs[1]:
        provedor = st.selectbox('Select the LLM', CONFIG_MODELOS.keys())
        modelo = st.selectbox('Select the model', CONFIG_MODELOS[provedor]['modelos'])
        api_key = st.text_input(
            f'Add the API key for the LLM provider: {provedor}',
            value=st.session_state.get(f'api_key_{provedor}')
            )

        st.session_state[f'api_key_{provedor}'] = api_key
    
    if st.button('Load OracleFlow', use_container_width=True):
        carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)
    
    if st.button('Clean OracleFlow History', use_container_width=True):
        st.session_state['memoria'] = MEMORIA

def main():
    with st.sidebar:
        sidebar()
    pagina_chat()

if __name__ == '__main__':
    main()