import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()

st.title("개념기반 탐구학습 질문 생성💭")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "chain" not in st.session_state:
    st.session_state["chain"] = None

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    task_input = st.text_input("성취기준 또는 학습 주제 입력", "")
    submit_button = st.button(label="질문 생성")

if clear_btn:
    st.session_state["messages"] = []
    st.session_state["chain"] = None


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메세지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


def create_prompt(task_input):
    prompt_template = f"""
    {task_input}에 성취기준 또는 학습 주제를 입력하면 그것을 참고하여 아래 예시 질문과 비슷한 핵심 질문, 출발 질문, 전개 질문, 도착 질문을 표로 만들어주세요.

    핵심 질문은 학습 목표와 학습 주제에 대한 깊이있는 교사의 고민이 결합되어야 나올 수 있는 질문이며 지식 뿐 아니라 기능이나 태도, 가치 등이 핵심 질문이 될 수 있다.   
    출발 질문은 도입 부분에 해당하는 질문으로, 학습 주제와 학생들의 실생활을 연결하여 흥미와 지적 호기심을 자극하고 학생들의 참여를 이끌어냅니다.  
    전개 질문은 본 시 학습에서 다루는 학습 내용에 관한 질문으로, 주로 지식과 이해를 요구하는 수렴적이고 닫힌 형태의 질문입니다.  
    도착 질문은 배운 지식과 실제 삶을 연결하는 질문으로, 적용, 분석, 종합, 평가에 해당하는 발산적이고 열린 형태의 질문입니다.

    예시:
    * 성취기준 또는 학습 주제: 화산과 지진
    1. 핵심 질문:
        - "만약 우리나라에서 화산과 지진 현상이 일어난다면 어떻게 하면 좋을까?"
    2. 출발 질문:
        - "만약 백두산이나 후지산에서 화산 활동이 일어난다면 어떤 일이 생길까?"
        - "2017년 포항에서 지진이 일어났을 때 많은 포항 사람들이 고통을 받았는데 우리 동네에 포항 지진처럼 강한 지진이 일어난다면?"
    3. 전개 질문:
        - "화산 활동과 지진이 일어나는 이유는?"
        - "지진의 세기 중 규모와 진도는 어떻게 다를까?"
        - "화산 활동과 지진은 어디에서 자주 발생할까?"
    4. 도착 질문:
        - "많은 과학자가 조만간 백두산이 화산 활동을 할 가능성이 있다고 경고하는데, 그렇다면 내가 대통령이라면 이 문제를 어떻게 해결하면 좋을까?"
        - "우리 동네에서 지진이 일어날 경우를 가정하고, 우리 모둠에서 지진 대피 매뉴얼을 만든다면 어떻게 만들면 좋을까?"
    
    # Task:
    {task_input}

    # Answer:
    """
    return ChatPromptTemplate.from_template(prompt_template)


llm = ChatOpenAI(model_name="gpt-4o", temperature=0)


# 학년군 또는 학습 주제가 변경될 때 체인 재생성
def update_chain(task_input):
    prompt = create_prompt(task_input)
    chain = {"task": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    st.session_state["chain"] = chain


# 이전 대화 기록 출력
print_messages()

# 경고 메시지 출력을 위한 빈 영역
warning_msg = st.empty()

# 결과를 반환하는 버튼 클릭 시
if submit_button:
    if task_input:
        update_chain(task_input)
        chain = st.session_state["chain"]

        if chain is not None:
            user_input = f"{task_input}"
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
            warning_msg.warning(
                "체인이 초기화되지 않았습니다. 페이지를 새로고침해주세요."
            )
    else:
        warning_msg.warning("성취기준 또는 학습 주제를 입력해주세요.")
