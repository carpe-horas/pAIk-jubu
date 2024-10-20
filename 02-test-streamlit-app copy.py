# 챗 히스토리 + 각 답변받기 + 말풍선 구현

# pip install -r requirements.txt
# streamlit run 02-test-streamlit-app.py

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

# 더미 질문과 답변 데이터를 리스트로 생성
dummy_data = [
    {"question": "더미 질문 1: 김치찌개 요리법을 알려주세요.", "answer": "더미 응답 1: 김치찌개를 만들기 위해서는 먼저 돼지고기와 김치를 볶아야 합니다."},
    {"question": "더미 질문 2: 된장찌개는 어떻게 만들죠?", "answer": "더미 응답 2: 된장찌개를 만들기 위해서는 된장과 고추장을 풀고 야채를 넣어야 합니다."},
    {"question": "더미 질문 3: 계란말이는 어떻게 만드나요?", "answer": "더미 응답 3: 계란말이는 계란을 풀어서 잘 익힌 후 돌돌 말아 만듭니다."},
    {"question": "더미 질문 4: 잡채 만드는 법은?", "answer": "더미 응답 4: 잡채를 만들기 위해서는 당면을 삶고 다양한 야채를 볶아야 합니다."},
    {"question": "더미 질문 5: 갈비찜 만드는 법을 알려주세요.", "answer": "더미 응답 5: 갈비찜은 갈비를 삶고 양념을 넣어 푹 끓여야 맛있습니다."}
]

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
        background-color: #0084ff;
        color: white;
        align-self: flex-end;
    }
    
    /* 사용자 말풍선 꼬리 */
    .user-bubble::after {
        content: "";
        position: absolute;
        bottom: 0;
        right: -10px;
        width: 0;
        height: 0;
        border-left: 10px solid #0084ff;
        border-top: 10px solid transparent;
        border-bottom: 10px solid transparent;
    }

    /* AI 말풍선 */
    .assistant-bubble {
        background-color: #e4e6eb;
        color: black;
        align-self: flex-start;
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
        margin: 20px 0;
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

# api_key = os.getenv("OPENAI_API_KEY")
# if not api_key:
#     st.error("OpenAI API Key가 설정되지 않았습니다.")
# else:
#     st.write("OpenAI API Key가 성공적으로 로드되었습니다.")


# FAISS 벡터 스토어는 24-vectorstore-save.ipynb에서 저장한 모델을 사용합니다.
# 이 파일이 존재하고 있는 디렉토리 하위에 db/faiss 디렉토리에 벡터 스토어 DB 파일이 존재해야 합니다

# FAISS 벡터 스토어 로드 시 사용자에게 보여줄 메시지 수정
def load_faiss():
    # 로딩 중 메시지를 위한 placeholder 생성
    loading_message = st.empty()
    
    # 로딩 중 메시지 표시
    loading_message.markdown('<div style="text-align: center; font-size: 18px; color: #FFD700;">AI 모델을 불러오고 있습니다... 잠시만 기다려 주세요 😊</div>', unsafe_allow_html=True)
    
    # 실제 FAISS 벡터 스토어 로드
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = FAISS.load_local('./db/faiss', OpenAIEmbeddings(), allow_dangerous_deserialization=True)

    # 로드 완료 후 기존 로딩 메시지를 성공 메시지로 변경
    loading_message.markdown('<div style="text-align: center; font-size: 18px; color: #FFD700;">AI 모델이 성공적으로 로드되었습니다! 🤖🍳</div>', unsafe_allow_html=True)
    
    return vector_store



#기존 코드
# def load_faiss():
#     st.write("FAISS 벡터 스토어를 불러오는 중입니다...")    
#     embeddings = OpenAIEmbeddings(openai_api_key=api_key)
#     vector_store = FAISS.load_local('./db/faiss', OpenAIEmbeddings(), allow_dangerous_deserialization=True)
#     st.write("벡터 스토어를 로드했습니다.")
#     return vector_store

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

# st.session_state를 사용하여 더미 데이터의 현재 인덱스를 저장
if "dummy_index" not in st.session_state:
    st.session_state["dummy_index"] = 0

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
        # 응답 음성 듣기 버튼에 고유 키 추가 (채팅 히스토리 길이와 index를 함께 사용)
        if index is not None:
            unique_key = f"listen_button_{index}_{len(st.session_state['chat_history'])}"
            if st.button(f"🔊", key=unique_key):
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
    

    ###############
    # 답변이 너무 오래 걸려서 사용한 openapi
    from langchain.chat_models import ChatOpenAI  # OpenAI API를 사용한 LLM

    llm = ChatOpenAI(openai_api_key=api_key, temperature=0.5)  # OpenAI API를 호출하는 부분
    retriever = vector_store.as_retriever()
    rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    result = rag_chain.invoke(query)
    ################


    # from langchain_community.chat_models import ChatOllama
    # from langchain_core.output_parsers import StrOutputParser
    # from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    # from langchain_core.callbacks.manager import CallbackManager

    # llm = ChatOllama(model="llama2:7b", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    # retriever = vector_store.as_retriever()
    # rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    # result = rag_chain.invoke(query)

    # 새로운 대화 내용을 세션 상태에 저장 (사용자 입력 및 AI 응답)
    st.session_state["chat_history"].append({"role": "user", "message": query})
    st.session_state["chat_history"].append({"role": "assistant", "message": result})

    # 사용자 및 AI 응답 메시지 표시
    display_chat_history()

# 더미 데이터로 대화 테스트 처리 함수
def process_dummy_data():
    # 현재 인덱스에 해당하는 더미 질문과 응답을 가져옴
    if st.session_state["dummy_index"] < len(dummy_data):
        data = dummy_data[st.session_state["dummy_index"]]
        # 중복된 질문이나 응답이 기록되지 않도록 확인
        if not any(entry['message'] == data['question'] for entry in st.session_state["chat_history"]):
            st.session_state["chat_history"].append({"role": "user", "message": data["question"]})
        if not any(entry['message'] == data['answer'] for entry in st.session_state["chat_history"]):
            st.session_state["chat_history"].append({"role": "assistant", "message": data["answer"]})
        # 인덱스를 증가시켜 다음 질문으로 넘어가도록 함
        st.session_state["dummy_index"] += 1
    
    # 채팅 히스토리 표시
    display_chat_history()


# 챗봇 대화 기록 출력
display_chat_history()

# 벡터 스토어 설정
vector_store = load_faiss()

st.write("저는 요리를 도와 드리는 요리사 비서입니다.")

# 사용자 질문 입력
user_input = st.text_input("궁금한 요리에 대해 물어보세요.")

# 질문 처리 버튼
if st.button("질문 하기"):
    process_user_input(user_input, vector_store)

# 더미 데이터로 테스트하기 버튼
if st.button("더미 데이터로 테스트하기"):
    process_dummy_data()