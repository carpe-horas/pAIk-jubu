# pip install -r requirements.txt
# streamlit run 01-test-streamlit-app.py

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
import pyttsx3 


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
    
    /* 텍스트 색상 */
    .stTextInput, .stButton, .stAlert, .stMarkdown {
        color: white !important;
    }

    /* 버튼 색상 변경 */
    .stButton button, .disabled-button {
        background-color: #444;
        color: white;
        border-radius: 5px;
        height: 50px;  /* 버튼 높이 설정 */
        font-size: 16px;
    }

    /* 비활성화된 버튼 스타일 */
    .disabled-button {
        background-color: #777;  
        color: white;
        border-radius: 5px;
        opacity: 0.6;
        pointer-events: none;
    }

    /* 텍스트 입력 상자 배경 및 텍스트 색상 */
    .stTextInput input {
        background-color: #333;
        color: white;
        height: 50px;
    }

    /* 페이지 제목 색상 변경 */
    h1 {
        color: #FFD700 !important; 
    }

    /* 텍스트 입력 설명 색상 변경 */
    .stTextInput label {
        color: #FFD700 !important; 
    }

    .full-width-text {
        width: 600%;  
        color: white;  
        font-size: 16px;  
        padding: 10px 0;
    }
    .warning-text {
        width: 300%;
        color: red;
        font-size: 16px;
        padding: 10px;
        border-radius: 5px;

    /* 채팅 스타일 커스텀 */
    .stChatMessage {
        background-color: #222 !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 10px !important;
        margin-bottom: 10px !important;
        width: 100% !important;  /* 채팅 메시지의 너비를 강제 설정 */
        display: inline-block !important;
        word-wrap: break-word !important;
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

# 제목을 HTML로 커스텀하여 표시
st.markdown("<h1>pAIk 주부 요리 비서</h1>", unsafe_allow_html=True)

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

# FAISS 벡터 스토어는 24-vectorstore-save.ipynb에서 저장한 모델을 사용합니다.
# 이 파일이 존재하고 있는 디렉토리 하위에 db/faiss 디렉토리에 벡터 스토어 DB 파일이 존재해야 합니다

def load_faiss():
    st.write("FAISS 벡터 스토어를 불러오는 중입니다...")    
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = FAISS.load_local('./db/faiss', OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    st.write("벡터 스토어를 로드했습니다.")
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

# 텍스트를 음성으로 변환(pyttsx3 라이브러리 사용)
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# 사용자 입력 처리
def process_user_input(query, vector_store):
    if not query:
        st.markdown('<div class="warning-text">질문을 입력해주세요.</div>', unsafe_allow_html=True)
        return
    
    st.markdown('<div class="full-width-text">벡터 스토어 검색 결과를 반영하여 답변을 준비하고 있습니다...</div>', unsafe_allow_html=True)
        
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
    
    from langchain_community.chat_models import ChatOllama
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from langchain_core.callbacks.manager import CallbackManager

    llm = ChatOllama(model="llama2:7b", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    retriever = vector_store.as_retriever()
    rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    result = rag_chain.invoke(query)

    # 답변을 st.session_state에 저장
    st.session_state["response"] = result

    display_chat_message("user", query)

    # AI 응답 메시지 표시
    display_chat_message("assistant", result)


# 벡터 스토어 설정
vector_store = load_faiss()

st.write("저는 요리를 도와 드리는 요리사 비서입니다.")

# 레이아웃을 나누어 텍스트 입력 상자와 녹음 버튼을 같은 행에 배치
col1, col2 = st.columns([4, 1])  # 4:1 비율로 레이아웃 나누기

# 음성 녹음 및 인식 기능
if "recording" not in st.session_state:
    st.session_state["recording"] = False
if "recognized_text" not in st.session_state:
    st.session_state["recognized_text"] = ""

# 사용자 질문 입력 (텍스트와 음성 입력)
with col1:
    user_input = st.text_input("궁금한 요리에 대해 물어보세요.", value=st.session_state["recognized_text"])


## 녹음 중단 버튼 없이 자동 중단
# 녹음 시작 버튼 눌러 음성 인식
with col2:
    st.markdown("<div style='padding-bottom: 30px;'></div>", unsafe_allow_html=True)  # 패딩을 추가하여 버튼을 내림
    if st.session_state["recording"]:
        st.session_state["recognized_text"] = recognize_speech()  # 녹음 시작 즉시 음성 인식
        st.session_state["recording"] = False
        st.experimental_rerun()  # 음성 인식 후 페이지 새로고침

    else:
        if st.button("🎤 녹음 시작"):
            st.session_state["recording"] = True
            st.experimental_rerun()  # 녹음 시작 상태 반영을 위한 새로고침


# 녹음 시작 및 중단 버튼
# 녹음 시작 버튼 누르고 녹음 중단 버튼을 누른 다음 녹음이 되는 오류가 있어서 위 중단 버튼 없는 위 코드 사용.
# with col2:
#     st.markdown("<div style='padding-bottom: 30px;'></div>", unsafe_allow_html=True)  # 패딩을 추가하여 버튼을 내림
#     if st.session_state["recording"]:
#         if st.button("🛑 녹음 중단"):
#             st.session_state["recording"] = False
#             st.session_state["recognized_text"] = recognize_speech()
#             st.experimental_rerun()  # 녹음이 끝난 후 페이지 새로고침
#     else:
#         if st.button("🎤 녹음 시작"):
#             st.session_state["recording"] = True
#             st.experimental_rerun()  # 녹음 시작 버튼을 누르면 상태를 새로고침하여 바로 반영

# 채팅 메시지 스타일링
def display_chat_message(role, message):
    if role == "user":
        st.markdown(f"""
            <div style="width: 600%; background-color: #999; color: #FFF; padding: 10px; border-radius: 10px; margin-bottom: 10px;">
                🧑 {message}
            </div>
        """, unsafe_allow_html=True)
    elif role == "assistant":
        st.markdown(f"""
            <div style="width: 600%; background-color: #FFF; color: #555; padding: 10px; border-radius: 10px; margin-bottom: 10px;">
                🥣 {message}
            </div>
        """, unsafe_allow_html=True)

    # 메시지 정렬 후 화면 갱신을 위해 추가
    st.markdown('<div style="clear: both;"></div>', unsafe_allow_html=True)

####### 더미 질문/응답 데이터 ########
#사용자 입력 처리 함수
def process_user_input(query, vector_store):
    # 질문이 없을 경우 더미 질문을 사용
    if not query:
        query = "더미 질문입니다. 이 질문은 UI 테스트용입니다."

    # 더미 응답 데이터를 세션 상태에 저장
    st.session_state["response"] = "더미 응답입니다. 이 응답은 UI 테스트용입니다. 요리법은 천차만별로 맛이 없을 수도 있습니다."

    # 사용자 메시지 표시
    display_chat_message("user", query)

    # AI 응답 메시지 표시
    display_chat_message("assistant", st.session_state["response"])
##################

# 질문 처리 버튼과 음성 듣기 버튼을 같은 행에 배치
col3, col4 = st.columns([1, 5])

# 질문 처리 버튼
with col3:
    if st.button("질문 하기"):
        process_user_input(user_input, vector_store)

# 답변을 음성으로 읽어주는 버튼
with col4:
    if "response" in st.session_state and st.session_state["response"]:
        # 답변이 있으면 정상 버튼 표시
        st.button("답변 음성으로 듣기", on_click=lambda: speak_text(st.session_state["response"]))
    else:
        # 답변이 없으면 비활성화된 것처럼 보이게 설정
        st.markdown('<button class="disabled-button">답변 음성으로 듣기</button>', unsafe_allow_html=True)

