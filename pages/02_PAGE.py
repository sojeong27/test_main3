from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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
# 문서에 포함되어 있는 정보를 검색하고 생성합니다.
retriever = vectorstore.as_retriever()

# 단계 6: 프롬프트 생성(Create Prompt)
# 프롬프트를 생성합니다.
prompt = PromptTemplate.from_template(
    """초등학교 3~4학년의 (1) 힘과 우리 생활 단원의 성취기준은 다음과 같습니다. "[4과01-01] 일상생활에서 힘과 관련된 현상에 흥미를 갖고, 물체를 밀거나 당길 때 나타나는 현상을 관찰할 수 있다. [4과01-02] 수평잡기 활동을 통해 물체의 무게를 비교할 수 있다. [4과01-03] 무게를 정확히 비교하기 위해서는 저울이 필요함을 알고, 저울을 사용해 무게를 비교할 수 있다. [4과01-04] 지레, 빗면과 같은 도구를 이용하면 물체를 들어 올릴 때 드는 힘의 크기가 달라짐을 알고, 도구가 일상생활에서 어떻게 쓰이는지 조사하여 공유할 수 있다." 이러한 내용을 참고하여 성취기준을 찾는 방법을 알고, 성취기준을 수정하지 말고, 그대로 모두 찾아서 알려주세요.

#Question: 
{question} 
#Context: 
{context} 

#Answer:"""
)

# 단계 7: 언어모델(LLM) 생성
# 모델(LLM) 을 생성합니다.
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# 단계 8: 체인(Chain) 생성
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
