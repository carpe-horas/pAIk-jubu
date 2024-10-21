# pip install -r requirements.txt
# streamlit run 06-test-streamlit-app.py
# css - 애니메이션 적용

import os
import streamlit as st
import time
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI

from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
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
    <link href="https://fonts.googleapis.com/css2?family=Jua&display=swap" rel="stylesheet">

    <style>
    /* 페이지 전체 배경 색상 */
    .main {
        background-color: #FFF8E1;  
    }

    /* AI 모델 로드 텍스트 */
    .info-text {
        color: #FF8C69;  
        font-size: 20px;
        margin-bottom: 15px;
        text-align: center;
        font-family: 'Jua', sans-serif; 
        opacity: 1;
        animation: fadeOut 5s forwards; /* 5초 동안 표시 후 사라지는 애니메이션 */
    }

     /* 애니메이션 정의 */
    @keyframes fadeOut {
        0% {
            opacity: 1; /* 처음에는 완전히 보임 */
        }
        80% {
            opacity: 1; /* 4초 동안 유지 */
        }
        100% {
            opacity: 0; /* 5초 후에 완전히 사라짐 */
        }
    }

    /*인풋창 위 텍스트*/
    .question-text{
        color: #FF7043;  
        font-size: 20px;
        margin-top: 5px;
        margin-bottom: 2px;
        text-align: center;
        font-family: 'Jua', sans-serif; 
    }

    /* 페이지 제목 스타일 */
    h1 {
        color: #FF5722;  
        text-align: center;
        font-family: 'Jua', sans-serif; 
        font-size: 40px;  
        font-weight: bold;  
        margin-bottom: 40px;
        animation: glow 1.3s ease-in-out infinite alternate; /* 애니메이션 적용 */
    }

    /* 글자에 애니메이션 효과 추가 */
    @keyframes glow {
        from {
            text-shadow: 0 0 10px #FF7043, 0 0 20px #FF7043, 0 0 30px #FF7043;
            color: #FF5722;
        }
        to {
            text-shadow: 0 0 20px #FF8C69, 0 0 30px #FF8C69, 0 0 40px #FF8C69;
            color: #FF8C69;
        }
    }

    /* 채팅 컨테이너 */
    .chat-container {
        display: flex;
        flex-direction: column;
        width: 100%;
        margin-top: 20px;
    }

    /* 말풍선 공통 스타일 */
    .chat-bubble {
        padding: 15px;
        border-radius: 15px;
        margin-bottom: 10px;
        word-wrap: break-word;
        position: relative;
        display: inline-block;
        font-family: 'Jua', sans-serif; 
        font-size: 16px;
        color: #5A3D05;
        line-height: 1.5;
        max-width: 60%;  
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);  /* 부드러운 그림자 */
    }

    /* 사용자 메시지 (오른쪽 정렬) */
    .user-bubble {
        background-color: #F7E269;  
        color: #822903;
        float: right;  /* 말풍선을 오른쪽으로 배치 */
        border-radius: 15px 15px 0px 15px;  /* 말풍선 모서리 */
        text-align: left;
        width: auto;  
        max-width: 60%;  
        margin-top: 10px;
        margin-right: 20px;  
    }

    /* 사용자 말풍선 꼬리 */
    .user-bubble::after {
        content: "";
        position: absolute;
        bottom: 0;
        right: -10px;
        width: 0;
        height: 0;
        border-left: 10px solid #F7E269;
        border-top: 10px solid transparent;
        border-bottom: 10px solid transparent;
    }

    /* AI 메시지 (왼쪽 정렬) */
    .assistant-bubble {
        background-color: #FFC044;  
        color: #822903;
        float: left;  
        border-radius: 15px 15px 15px 0px;
        text-align: left;
        width: auto;  
        max-width: 60%;  
        margin-bottom: -25px;
    }

    /* AI 말풍선 꼬리 */
    .assistant-bubble::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: -10px;
        width: 0;
        height: 0;
        border-right: 10px solid #FFC044;
        border-top: 10px solid transparent;
        border-bottom: 10px solid transparent;
    }

    /* 입력창과 버튼 스타일 */
    .stTextInput input {
        background-color: #FFF3E0;  
        color: #822903;
        border-radius: 10px;
        border: 1px solid #FF7043;
        padding: 10px;
    }

    stTextInput input:hover {
        background-color: #F4834F;
        color: #f5c4b8; 
    }

    /* 버튼 스타일 */
    .stButton button {
        background-color: #FF7043;  
        color: white;
        border-radius: 10px;
        font-size: 16px;
        height: 45px;
        width: 135px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        margin: 2px;  /* 버튼 간의 간격 */
        margin-top: 27px; 
        padding: 0px 20px; 
    }

    .stButton button:hover {
        background-color: #F4834F;
        color: #f5c4b8; 
    }

    /* 두 버튼 간 간격을 줄이기 위해 추가 */
    .stButton + .stButton {
        margin-left: 5px;  
    }

    /* 스크롤바 색상 */
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #FFF8E1;
    }
    ::-webkit-scrollbar-thumb {
        background: #FF7043;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #E64A19;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

st.markdown("<h1>👩‍🍳 pAIk 주부 요리 비서 </h1>", unsafe_allow_html=True)

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
    loading_message.markdown('<div class="info-text">AI 모델을 불러오고 있습니다... 잠시만 기다려 주세요 😊</div>', unsafe_allow_html=True)
    
    # 실제 FAISS 벡터 스토어 로드
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = FAISS.load_local('./db/faiss', OpenAIEmbeddings(), allow_dangerous_deserialization=True)

    # 로드 완료 후 기존 로딩 메시지를 성공 메시지로 변경
    loading_message.markdown('<div class="info-text">AI 모델이 성공적으로 로드되었습니다!</div>', unsafe_allow_html=True)
    
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
    try:
        # 음성을 gTTS로 변환
        tts = gTTS(text=text, lang='ko')
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)

        # 음성을 브라우저에서 재생하기 위해 base64로 인코딩
        audio_data = base64.b64encode(mp3_fp.read()).decode("utf-8")
        audio_html = f'<audio autoplay="true" controls><source src="data:audio/mp3;base64,{audio_data}" type="audio/mp3"></audio>'
        
        st.markdown(audio_html, unsafe_allow_html=True)

    except Exception as e:
        # 오류 발생 시 사용자에게 알림
        st.error(f"음성 변환 중 오류가 발생했습니다: {e}")

# 채팅 메시지를 session_state에 저장하여 메모리 기능 구현
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# 애니메이션 상태를 저장하기 위한 session state 확인 및 초기화
if "animation_finished" not in st.session_state:
    st.session_state["animation_finished"] = {}

# 채팅 메시지 스타일링 (타이핑 애니메이션 적용)
def display_chat_message(role, message, index=None):
    # 해당 메시지에 대해 애니메이션이 이미 끝났는지 확인
    message_key = hash(message)  # 메시지의 해시 값을 키로 사용

    if message_key not in st.session_state["animation_finished"]:
        # 말풍선을 위한 placeholder 생성
        bubble_placeholder = st.empty()  
        
        # 타이핑 애니메이션 적용
        typing_animation(message, role, bubble_placeholder)
        
        # 애니메이션이 완료되었음을 기록
        st.session_state["animation_finished"][message_key] = True
        
        # 애니메이션이 끝난 후에도 메시지를 고정적으로 표시
        bubble_placeholder.markdown(f'<div class="chat-bubble {role}-bubble">{message}</div>', unsafe_allow_html=True)
    else:
        # 이미 애니메이션이 끝난 메시지는 고정적으로 표시
        st.markdown(f'<div class="chat-bubble {role}-bubble">{message}</div>', unsafe_allow_html=True)

    # AI의 경우 '듣기' 버튼 추가
    if role == "assistant" and index is not None:
        unique_key = f"listen_button_{index}_{hash(message)}"
        if st.button("🔊 듣기", key=unique_key):
            speak_text_gtts(message)

# 타이핑 애니메이션 함수
def typing_animation(message, role, bubble_placeholder, delay=0.05):
    displayed_text = ""
    for char in message:
        displayed_text += char
        bubble_placeholder.markdown(f'<div class="chat-bubble {role}-bubble">{displayed_text}</div>', unsafe_allow_html=True)
        time.sleep(delay)  # 글자가 한 글자씩 나타나는 딜레이 적용

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
    from langchain.chat_models import ChatOpenAI 

    llm = ChatOpenAI(openai_api_key=api_key, temperature=0.5)
    retriever = vector_store.as_retriever()
    rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    result = rag_chain.invoke(query)
    ################


    # from langchain_community.chat_models import ChatOllama
    # from langchain_core.output_parsers import StrOutputParser

    # llm = ChatOllama(model="llama2:7b", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    # retriever = vector_store.as_retriever()
    # rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    # result = rag_chain.invoke(query)

    # 중복된 답변이 이미 있는지 확인 후 추가
    if not any(entry['message'] == result and entry['role'] == "assistant" for entry in st.session_state["chat_history"]):
        st.session_state["chat_history"].append({"role": "assistant", "message": result})


# 벡터 스토어 설정
vector_store = load_faiss()

st.markdown('<div class="question-text">저는 요리를 도와 드리는 요리사 비서입니다.</div>', unsafe_allow_html=True)

st.markdown('<div class="question-text">궁금한 요리에 대해 물어보세요.</div>', unsafe_allow_html=True)

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