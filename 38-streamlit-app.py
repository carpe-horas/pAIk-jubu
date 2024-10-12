#pip install -r requirements.txt
#streamlit run c:/Users/carpe/OneDrive/Desktop/p/rag/38-streamlit-app.py

###
# 이 프로그램은 OpenAIEmbedding을 이용하여 만개의 레시피 사이트에서 수집한 5000여개의 레시피 정보를
# csv 포맷으로 정리하여 FAISS 벡터스토어에 저장한 벡터스토어 데이터를 읽어와서 요리 정보를 제공하는 RAG
# 어플레케이션으로 streamlit 웹 인터페이스로 제공할 수 있도록 만들었습니다.
# 
#
import os
import streamlit as st
from dotenv import load_dotenv
#from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI



from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 로컬 환경에서 실행할 때 이 실행파일이 존재하는 디렉토리에 .env 라는 이름의 파일을 생성하여 ChatGPT 사용을 위한 API KEY와 같은
# 환경 변수 설정을 가져와서 사용할 수 있도록 하는 코드입니다.abs
# .env 파일 내용은 아래와 같이 미리 정의해 두어야 합니다. 
# OPENAI_API_KEY="sk-proj-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
#
# 환경 변수 로드
load_dotenv()

# OpenAI API 키 로드
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API Key가 설정되지 않았습니다.")
else:
    st.write("OpenAI API Key가 성공적으로 로드되었습니다.")

# FIASS 백터 스토어는 24-vectorstore-save.ipynb에서 저장한 모델을 사용합니다.
# 이 파일이 존재하고 있는 디렉토리 하위에 db/faiss 디렉토리에 벡터 스토어 DB 파일이 존재해야 합니다
# MacOS의 경우 아래와 같은 형식으로 보입니다
# $ ls -l db/faiss
# total 84080
# -rw-r--r--@ 1 khuh  staff  35702829 10  9 14:55 index.faiss
# -rw-r--r--@ 1 khuh  staff   5752313 10  9 14:55 index.pkl

# FAISS 벡터 스토어 로드
# 벡터스토어가 로컬에 저장되어 있어도 로드할 때 OpenAIEmbedding을 사용하기 때문에 아직 OpenAI API 키가 필요합니다.
# ToDo: OpenAIEmbedding 대신 HuggingFace의 Embedding을 사용할 수 있도록 변경할 예정입니다

def load_faiss():
    st.write("FAISS 벡터 스토어를 불러오는 중입니다...")    
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    # 저장된 벡터스토어 로드
    vector_store = FAISS.load_local('./db/faiss', OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    
    st.write("벡터 스토어를 로드했습니다.")
   
    return vector_store

# 사용자 입력 처리
def process_user_input(query, vector_store):
    if not query:
        st.warning("질문을 입력해주세요.")
        return
    
    st.write("벡터 스토어 검색 결과를 반영하여 답변을 준비하고 있습니다...")
    
    # 프롬프트 템플릿 설정
    # 요리 조리법을 보기 쉬은 포맷으로 출력할 수 있는도록 프롬프트 답변 방식을 지시합니다. 
    from langchain_core.prompts import PromptTemplate

    prompt = PromptTemplate.from_template(
        """당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 
        당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
        검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 
        만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다` 라고 답하세요.
        한국어로 답변해 주세요. 문맥(context)에 재료가 있는 경우 목록으로 표시해주세요. 
        문맥(context)에 url이 있으면 답변에 `출처: url\n` 형식으로 표시해주세요. 
        단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.

#Question:
{question}

#Context:
{context}

#Answer:"""
)
    
    # Ollama에서 Qwen2.5 모델을 LLM으로 사용합니다. 
    # 로컬 시스템에 ollama 가 설치되어 있어야 합니다. ollama가 실행중인지 확인은 `ps aux | grep ollama` 명령어로 확인할 수 있습니다.
    # 윈도우에서 확인은 Get-Process | Where-Object { $_.Name -like "*ollama*" }

    # ollama list 명령으로 사용하려고 하는 LLM 모델이 있어야 합니다.
    # 여기에서 사용한 모델은 아래 링크에서 가져왔습니다. 
    # https://huggingface.co/teddylee777/Qwen2.5-7B-Instruct-kowiki-qa-context-gguf
    from langchain_community.chat_models import ChatOllama
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from langchain_core.callbacks.manager import CallbackManager

    llm = ChatOllama(
        #model="qwen2.5-7b-instruct-kowiki:latest",
        model="llama2:7b", 
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    
    # Retriever를 사용하여 문맥(context) 에서 질문(question) 을 찾습니다.
    retriever = vector_store.as_retriever()
    
    # RAG Chain을 구성합니다.
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
  
    # 질문을 처리하여 결과를 반환합니다.
    result = rag_chain.invoke(query)
    with st.chat_message("user"):
        st.write(query)
        
    with st.chat_message("assistant"):
        st.write(result)
    # 디버깅용 코드 질문에 대해 retriever 에서 답변을 찾아 터미널에 출력합니다. 실제 응답과 비교하여 RAG가 잘 작동하는지 확인합니다. 
    #query = "토마토"
    #results = retriever.get_relevant_documents(query)
    ## 검색 결과 출력
    #for result in results:
    #    print(result.page_content)    

# Streamlit 앱 UI 구성
st.title("pAIk 주부 요리 비서")

# 벡터 스토어 설정
vector_store = load_faiss()

st.write("저는 요리를 도와 드리는 요리사 비서입니다.")

# 사용자 질문 입력
user_input = st.text_input("궁금한 요리에 대해 물어보세요.", "")

# 질문 처리 버튼
if st.button("질문 하기"):
    process_user_input(user_input, vector_store)

