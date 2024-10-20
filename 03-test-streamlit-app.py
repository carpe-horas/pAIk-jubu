# pip install -r requirements.txt
# streamlit run 03-test-streamlit-app.py

import os
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI

from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import base64

# Streamlit 페이지 설정
st.set_page_config(page_title="pAIk 주부 요리 비서", layout="centered", initial_sidebar_state="collapsed")

# Streamlit UI를 커스텀하기 위해 HTML과 CSS 사용
st.markdown(
    """
    <style>
    /* 페이지 전체 배경 색상 */
    .main {
        background-color: black;
    }

    /* 입력창과 버튼을 나란히 붙이기 위한 스타일 */
    .input-container {
        display: flex;
        justify-content: flex-start;  /* 버튼과 입력창 정렬 */
        align-items: center; /* 가운데 정렬 */
    }

    /* 텍스트 색상 */
    .stTextInput, .stButton, .stAlert, .stMarkdown {
        color: white !important;
    }

    /* 버튼 색상 변경 */
    .stButton button {
        background-color: #444;
        color: white;
        border-radius: 5px;
        height: 45px;  /* 버튼 높이 설정 */
        width: auto;  /* 버튼의 너비를 자동으로 설정 */
        font-size: 16px;
        margin: 5px;  /* 버튼 간의 간격 추가 */
        margin-top: 25px; /* 버튼을 약간 아래로 이동 */
    }

    /* 텍스트 입력 상자 배경 및 텍스트 색상 */
    .stTextInput input {
        background-color: #333;
        color: white;
        height: 50px;
        margin-right: 5px; /* 입력창 오른쪽 여백 조정 */
    }

    /* 페이지 제목 색상 변경 */
    h1 {
        color: #FFD700 !important; 
    }

    /* 말풍선 스타일 */
    .chat-bubble {
        max-width: 60%;
        padding: 10px;
        border-radius: 15px;
        margin-bottom: 10px;
        word-wrap: break-word;
        position: relative;
        display: inline-block;
    }

    /* 사용자 말풍선 */
    .user-bubble {
        background-color: #f0edc5;
        color: #2e2c11;
        align-self: flex-end; /* 오른쪽 정렬 */
    }

    /* 사용자 말풍선 꼬리 */
    .user-bubble::after {
        content: "";
        position: absolute;
        bottom: 0;
        right: -10px;
        width: 0;
        height: 0;
        border-left: 10px solid #f0edc5;
        border-top: 10px solid transparent;
        border-bottom: 10px solid transparent;
    }

    /* AI 말풍선 */
    .assistant-bubble {
        background-color: #e4e6eb;
        color: black;
        align-self: flex-start; /* 왼쪽 정렬 */
    }

    /* AI 말풍선 꼬리 */
    .assistant-bubble::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: -10px;
        width: 0;
        height: 0;
        border-right: 10px solid #e4e6eb;
        border-top: 10px solid transparent;
        border-bottom: 10px solid transparent;
    }

    /* 채팅 영역 레이아웃 설정 */
    .chat-container {
        display: flex;
        flex-direction: column;
        margin: 0px 0;
    }

    /* 스크롤바 색상 */
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #333;
    }
    ::-webkit-scrollbar-thumb {
        background: #888;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    </style>
    """, 
    unsafe_allow_html=True
)


st.markdown("<h1>pAIk 주부 요리 비서</h1>", unsafe_allow_html=True)

# 로컬 환경에서 실행할 때 이 실행파일이 존재하는 디렉토리에 .env 라는 이름의 파일을 생성하여 ChatGPT 사용을 위한 API KEY와 같은
# 환경 변수 설정을 가져와서 사용할 수 있도록 하는 코드입니다.
# .env 파일 내용은 아래와 같이 미리 정의해 두어야 합니다. 
# OPENAI_API_KEY="sk-proj-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
#
# 환경 변수 로드
load_dotenv()

# OpenAI API 키 로드
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API Key가 설정되지 않았습니다.")
    st.stop()  # API 키가 없으면 더 이상 실행되지 않도록 함


# FAISS 벡터 스토어는 24-vectorstore-save.ipynb에서 저장한 모델을 사용합니다.
# 이 파일이 존재하고 있는 디렉토리 하위에 db/faiss 디렉토리에 벡터 스토어 DB 파일이 존재해야 합니다

# FAISS 벡터 스토어 로드 시 사용자에게 보여줄 메시지 수정
def load_faiss():
    # 로딩 중 메시지를 위한 placeholder 생성
    loading_message = st.empty()
    
    # 로딩 중 메시지 표시
    loading_message.markdown('<div style="text-align: left; font-size: 18px; color: #FFFFFF;">AI 모델을 불러오고 있습니다... 잠시만 기다려 주세요 😊</div>', unsafe_allow_html=True)
    
    # 실제 FAISS 벡터 스토어 로드
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = FAISS.load_local('./db/faiss', OpenAIEmbeddings(), allow_dangerous_deserialization=True)

    # 로드 완료 후 기존 로딩 메시지를 성공 메시지로 변경
    loading_message.markdown('<div style="text-align: left; font-size: 18px; color: #FFFFFF;">AI 모델이 성공적으로 로드되었습니다!</div>', unsafe_allow_html=True)
    
    return vector_store

# Google Speech Recognition을 사용한 음성 인식 함수
def recognize_speech():
    r = sr.Recognizer()
    
    # 마이크로부터 음성 입력 받기
    with sr.Microphone() as source:
        st.write("음성 인식중.....")
        r.adjust_for_ambient_noise(source)  # 주변 소음 조절
        audio = r.listen(source)  # 음성 입력 받기
    
    try:
        # Google Speech Recognition을 사용하여 음성을 텍스트로 변환
        text = r.recognize_google(audio, language='ko-KR')
        st.write("인식 완료")
        return text
    except sr.UnknownValueError:
        st.write("음성을 인식할 수 없습니다.")
        return ""
    except sr.RequestError as e:
        st.write(f"Google Speech Recognition 서비스에 문제가 발생했습니다: {e}")
        return ""

# 텍스트를 음성으로 변환하여 재생하는 함수 (gTTS 사용)
def speak_text_gtts(text):
    tts = gTTS(text=text, lang='ko')
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

    # 음성을 브라우저에서 재생하기 위해 base64로 인코딩
    audio_data = base64.b64encode(mp3_fp.read()).decode("utf-8")
    audio_html = f'<audio autoplay="true" controls><source src="data:audio/mp3;base64,{audio_data}" type="audio/mp3"></audio>'
    
    st.markdown(audio_html, unsafe_allow_html=True)

# 채팅 메시지를 session_state에 저장하여 메모리 기능 구현
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# 채팅 메시지 스타일링 (말풍선 모양 적용)
def display_chat_message(role, message, index=None):
    if role == "user":
        st.markdown(f"""
            <div class="chat-bubble user-bubble">
                🧑 {message}
            </div>
        """, unsafe_allow_html=True)
    elif role == "assistant":
        st.markdown(f"""
            <div class="chat-bubble assistant-bubble">
                🥣 {message}
            </div>
        """, unsafe_allow_html=True)
        
        # 고유한 키 생성: index, message의 해시값 사용
        if index is not None:
            unique_key = f"listen_button_{index}_{hash(message)}"
            if st.button("🔊 듣기", key=unique_key):
                speak_text_gtts(message)


# 챗봇 대화 히스토리 출력 함수
def display_chat_history():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for i, entry in enumerate(st.session_state["chat_history"]):
        display_chat_message(entry["role"], entry["message"], index=i if entry["role"] == "assistant" else None)
    st.markdown('</div>', unsafe_allow_html=True)

# 사용자 입력 처리 함수
def process_user_input(query, vector_store):
    if not query:
        st.markdown('<div class="warning-text">질문을 입력해주세요.</div>', unsafe_allow_html=True)
        return

    # 사용자 질문 추가
    if not any(entry['message'] == query and entry['role'] == "user" for entry in st.session_state["chat_history"]):
        st.session_state["chat_history"].append({"role": "user", "message": query})

        
    from langchain_core.prompts import PromptTemplate
    prompt = PromptTemplate.from_template(
        """당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 
        당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
        검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 
        한국어로만 답변해 주세요.

#Question:
{question}

#Context:
{context}

#Answer:"""
    )
    

    ###############
    # 답변이 너무 오래 걸려서 테스트시 사용한 openapi
    # from langchain.chat_models import ChatOpenAI 

    # llm = ChatOpenAI(openai_api_key=api_key, temperature=0.5)
    # retriever = vector_store.as_retriever()
    # rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    # result = rag_chain.invoke(query)
    ################


    from langchain_community.chat_models import ChatOllama
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from langchain_core.callbacks.manager import CallbackManager

    llm = ChatOllama(model="llama2:7b", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    retriever = vector_store.as_retriever()
    rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    result = rag_chain.invoke(query)

    # 중복된 답변이 이미 있는지 확인 후 추가
    if not any(entry['message'] == result and entry['role'] == "assistant" for entry in st.session_state["chat_history"]):
        st.session_state["chat_history"].append({"role": "assistant", "message": result})


    # 사용자 및 AI 응답 메시지 표시
    display_chat_history()

# 벡터 스토어 설정
vector_store = load_faiss()

st.markdown('<div style="text-align: left; font-size: 18px; color: #FFD7;">저는 요리를 도와 드리는 요리사 비서입니다.</div>', unsafe_allow_html=True)

st.markdown('<div style="text-align: left; font-size: 18px; color: #FFD700;">궁금한 요리에 대해 물어보세요.</div>', unsafe_allow_html=True)

# 입력창과 버튼을 나란히 배치
col1, col2, col3 = st.columns([3, 1, 1])  

with col1:
    user_input = st.text_input("")

with col2:
    submit_button = st.button("질문 하기")

with col3:
    speech_button = st.button("🎤 음성 질문")

# 버튼 동작 처리
if submit_button:
    process_user_input(user_input, vector_store)

if speech_button:
    user_input = recognize_speech()
    if user_input:
        process_user_input(user_input, vector_store)

# 챗봇 대화 기록 출력
display_chat_history()
