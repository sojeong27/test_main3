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
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_bnt = st.button("대화 초기화")

    selected_grade = st.selectbox(
        "학년군을 선택해주세요",
        ["초등학교 3~4학년", "초등학교 5~6학년", "중학교 1~3학년"],
        index=0,
    )
    task_input = st.text_input("학습 주제를 입력해주세요", "")

    # 초기화 버튼 클릭 시 대화 초기화
    if clear_bnt:
        st.session_state["messages"] = []

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
# 벡터스토어를 생성합니다.
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

# 단계 5: 검색기(Retriever) 생성
retriever = vectorstore.as_retriever()

# 단계 6: 프롬프트 생성(Create Prompt)
def create_prompt(selected_grade, task_input):
    prompt_template = f"""
    {selected_grade}의 학습 성취기준은 다음과 같습니다:
    "[4과01-01] 일상생활에서 힘과 관련된 현상에 흥미를 갖고, 물체를 밀거나 당길 때 나타나는 현상을 관찰할 수 있다.
     [4과01-02] 수평잡기 활동을 통해 물체의 무게를 비교할 수 있다.
     [4과01-03] 무게를 정확히 비교하기 위해서는 저울이 필요함을 알고, 저울을 사용해 무게를 비교할 수 있다.
     [4과01-04] 지레, 빗면과 같은 도구를 이용하면 물체를 들어 올릴 때 드는 힘의 크기가 달라짐을 알고, 도구가 일상생활에서 어떻게 쓰이는지 조사하여 공유할 수 있다."
    이러한 내용을 참고하여 성취기준을 찾는 방법을 알고, {task_input}와 관련된 성취기준을 수정하지 말고, 그대로 모두 찾아서 알려주세요.

    # Question:
    {task_input}
    # Context:
    {{context}}

    # Answer:
    """
    return ChatPromptTemplate.from_template(prompt_template)

# 단계 7: 언어모델(LLM) 생성
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# 단계 8: 체인(Chain) 생성 및 초기화
if "chain" not in st.session_state:
    if task_input:
        prompt = create_prompt(selected_grade, task_input)
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        st.session_state["chain"] = chain

# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 만약에 사용자 입력이 들어오면...
if user_input and "chain" in st.session_state:
    chain = st.session_state["chain"]

    if chain is not None:
        # 사용자의 입력
        st.chat_message("user").write(user_input)
        # 스트리밍 호출
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
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
