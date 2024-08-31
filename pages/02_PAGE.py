import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()

# Streamlit 프로젝트 제목 설정
st.title("교육과정 기반 QA📜")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "chain" not in st.session_state:
    st.session_state["chain"] = None

# 사이드바 생성 및 초기화
with st.sidebar:
    clear_btn = st.button("대화 초기화")

    selected_subject = st.selectbox(
        "교과를 선택해주세요",
        ["국어", "수학", "사회", "과학"],
        index=0,
    )

    selected_grade = st.selectbox(
        "학년군을 선택해주세요",
        ["초등학교 3~4학년", "초등학교 5~6학년"],
        index=0,
    )

    task_input = st.text_input("학습 주제를 입력해주세요", "")
    submit_button = st.button(label="성취기준 확인")

# 대화 초기화 버튼 클릭 시
if clear_btn:
    st.session_state["messages"] = []
    st.session_state["chain"] = None

# 이전 대화 기록 출력 함수
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

# 새로운 메시지를 세션 상태에 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

# 단계 1: 엑셀 문서 로드
try:
    loader = UnstructuredExcelLoader("./data/교육과정성취기준.xlsx", mode="elements")
    docs = loader.load()
except Exception as e:
    st.error(f"엑셀 파일을 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 단계 2: 문서 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)

# 단계 3: 임베딩 생성
embeddings = OpenAIEmbeddings()

# 단계 4: 벡터 저장소(DB) 생성
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

# 단계 5: 검색기 생성
retriever = vectorstore.as_retriever()

# 단계 6: 프롬프트 생성 함수
def create_prompt(selected_subject, selected_grade, task_input):
    prompt_template = f"""
    {selected_subject}에서 선택한 교과를 찾고 {selected_grade}에서 선택한 학년군을 찾은 다음 {task_input}와 관련된 성취기준을 찾아서 영역과 성취기준을 표로 만들어주세요.
    만약 {selected_subject}에서 선택한 교과와 {selected_grade}에서 선택한 학년군에서 {task_input}와 관련된 성취기준을 찾을 수 없다면 "해당 교과에서 관련된 성취기준을 찾을 수 없습니다."라고 말해주세요.
    
    # Task:
    {task_input}
    # Context:
    {{context}}

    # Answer:
    """
    return ChatPromptTemplate.from_template(prompt_template)

# 단계 7: 언어 모델 생성
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# 학년군 또는 학습 주제가 변경될 때 체인 재생성
def update_chain(selected_subject, selected_grade, task_input):
    prompt = create_prompt(selected_subject, selected_grade, task_input)
    chain = (
        {"context": retriever, "task": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    st.session_state["chain"] = chain

# 이전 대화 기록 출력
print_messages()

# 경고 메시지 출력을 위한 빈 영역
warning_msg = st.empty()

# 결과를 반환하는 버튼 클릭 시
if submit_button:
    if selected_subject and selected_grade and task_input:
        update_chain(selected_subject, selected_grade, task_input)
        chain = st.session_state["chain"]

        if chain is not None:
            user_input = f"{selected_subject}, {selected_grade}, {task_input}"
            # 사용자의 입력 출력
            st.chat_message("user").write(user_input)
            # 스트리밍 호출
            response = chain.stream(user_input)
            with st.chat_message("assistant"):
                container = st.empty()
                ai_answer = ""
                for token in response:
                    ai_answer += token
                    container.markdown(ai_answer)
            
            # 대화 기록을 세션 상태에 저장
            add_message("user", user_input)
            add_message("assistant", ai_answer)

        else:
            warning_msg.warning("체인이 초기화되지 않았습니다. 페이지를 새로고침해주세요.")
    else:
        warning_msg.warning("교과, 학년군, 학습 주제를 모두 입력해주세요.")
