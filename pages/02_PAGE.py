import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
st.title("교육과정 기반 QA📜")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "chain" not in st.session_state:
    st.session_state["chain"] = None

# 사이드바 생성 및 초기화
with st.sidebar:
    clear_bnt = st.button("대화 초기화")
    
    selected_grade = st.selectbox(
        "학년군을 선택해주세요",
        ["초등학교 3~4학년", "초등학교 5~6학년", "중학교 1~3학년"],
        index=0,
    )
    
    task_input = st.text_input("학습 주제를 입력해주세요", "")
    submit_button = st.button(label="성취기준 확인")

# 초기화 버튼 눌렀을 때 대화 초기화
if clear_bnt:
    st.session_state["messages"] = []
    st.session_state["chain"] = None

# 이전 대화 기록 출력 함수
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

# 새로운 메세지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

# 단계 1: 문서 로드(Load Documents)
loader = PyMuPDFLoader("data/과학과교육과정.pdf")
docs = loader.load()

# 단계 2: 문서 분할(Split Documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)

# 단계 3: 임베딩(Embedding) 생성
embeddings = OpenAIEmbeddings()

# 단계 4: DB 생성(Create DB) 및 저장
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

# 단계 5: 검색기(Retriever) 생성
retriever = vectorstore.as_retriever()

# 단계 6: 프롬프트 생성 함수
def create_prompt(selected_grade, task_input):
    prompt_template = f"""
    ### 지시사항:
    1. {selected_grade}가 "초등학교 3~4학년"이면, 성취기준 코드가 "4"로 시작하는 항목만 찾으세요.
    2. {selected_grade}가 "초등학교 5~6학년"이면, 성취기준 코드가 "6"으로 시작하는 항목만 찾으세요.
    3. {selected_grade}가 "중학교 1~3학년"이면, 성취기준 코드가 "9"로 시작하는 항목만 찾으세요.
    4. {selected_grade}에 맞는 학습 성취기준 코드를 찾고 {task_input}과 관련된 단원의 성취기준 코드와 내용을 모두 찾아 수정하지 말고, 성취기준 코드와 성취기준 내용을 표로 만들어주세요.
    
    ### 예시:
    "초등학교 3~4학년"에서 "(1) 힘과 우리 생활" 단원의 성취기준 코드와 내용은 다음과 같습니다:
    - [4과01-01] 일상생활에서 힘과 관련된 현상에 흥미를 갖고, 물체를 밀거나 당길 때 나타나는 현상을 관찰할 수 있다.
    - [4과01-02] 수평잡기 활동을 통해 물체의 무게를 비교할 수 있다.
    - [4과01-03] 무게를 정확히 비교하기 위해서는 저울이 필요함을 알고, 저울을 사용해 무게를 비교할 수 있다.
    - [4과01-04] 지레, 빗면과 같은 도구를 이용하면 물체를 들어 올릴 때 드는 힘의 크기가 달라짐을 알고, 도구가 일상생활에서 어떻게 쓰이는지 조사하여 공유할 수 있다.
    
    ### 출력 형식:
    | 성취기준 코드 | 성취기준 내용 |
    |---------------|----------------|
    | [코드]        | 내용           |
    | [코드]        | 내용           |
    {selected_grade}와 {task_input}에 맞는 성취기준을 찾아 표로 작성하세요.

    # Task:
    {task_input}
    # Context:
    {{context}}

    # Answer:
    """
    return ChatPromptTemplate.from_template(prompt_template)

# 단계 7: 언어모델(LLM) 생성
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# 단계 8: 체인(Chain) 생성 및 초기화
if task_input and selected_grade and st.session_state["chain"] is None:
    prompt = create_prompt(selected_grade, task_input)
    chain = (
        {"context": retriever, "task": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    st.session_state["chain"] = chain

# 이전 대화 기록 출력
print_messages()

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 결과를 반환하는 버튼 클릭 시
if submit_button:
    if selected_grade and task_input and "chain" in st.session_state:
        chain = st.session_state["chain"]

        if chain is not None:
            user_input = f"{selected_grade}, {task_input}"
            # 사용자의 입력
            st.chat_message("user").write(user_input)
            # 스트리밍 호출
            response = chain.stream(user_input)
            with st.chat_message("assistant"):
                container = st.empty()
                ai_answer = ""
                for token in response:
                    ai_answer += token
                    container.markdown(ai_answer)
        
            # 대화기록을 저장한다.
            add_message("user", user_input)
            add_message("assistant", ai_answer)

        else:
            warning_msg.warning("체인이 초기화되지 않았습니다. 페이지를 새로고침해주세요.")
    else:
        warning_msg.warning("학년군과 학습 주제를 모두 입력해주세요.")
