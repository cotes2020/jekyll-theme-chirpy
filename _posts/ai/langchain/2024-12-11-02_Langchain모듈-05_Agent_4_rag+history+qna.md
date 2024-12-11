---
title: [Langchain Study | 3-2. Agents 4 - Agent로 Q&A 챗봇 구현 (RunnableWithMessageHistory)]
categories: [AI, Langchain]
tags: [Langchain, AI, AI Application, Agent, Memory, RAG]		
---


RAG의 `retriver`과 `memory` 모듈이 포함된 Q&A 챗봇을 agent로 다시 만들어봅시다!

---

## 블로그 글 기반의 맥락 유지 Q&A 챗봇

[LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) 블로그의 포스팅된 글을 기반으로 Q&A 챗봇을 만들어 보겠습니다. 여기에서 기존 대화기록을 계속 유지하여서 맥락을 유지하는 기능도 추가해보겠습니다.

해당 어플리케이션의 아키텍쳐 이렇습니다.

![image.png]({{"/assets/img/posts/image.png"  | relative_url }}){width="712" height="614"}

* 대화의 질문/답변을 Chat history로 보관하고 있다가,
* 사용자에게 질문이 들어오면, Chat history를 이해한 질문을 재구성합니다.
  * 예를 들어,
  * 첫번째 질문이 "Task Decomposition에 대해 알려줘"이고, 그에 따른 답변을 받습니다.
  * 두번째 질문이 "그것이 많이 사용되는 예제를 알려줘"라고 했을 때,
    * LLM에 대화 기록과 두번째 질문을 주며, 대화 기록을 분석해서 두번째 질문에서 질문하는 바를 하나의 질문 그 자체만 보고도 알 수 있도록 재구성하라고 요청합니다.
    * 그동안의 대화 기록을 통해 `그것`이 `Task Decomposition`을 의미하는 것을 파악하여 `"Task Decomposition이 많이 사용되는 예제를 알려줘"`라는 질문이 생성됩니다.
    * "Task Decomposition이 많이 사용되는 예제를 알려줘"라는 질문을 통해 vector db에서 관련도 높은 문서를 찾아 반환합니다.
* 재구성된 질문을 가지고 다시 vectorDB에서 관련문서를 반환해서, 주어진 prompt를 가지고 llm에 질문하여 답을 받아옵니다.
* 위 과정을 다시 Chat history에 저장됩니다.

### normal 버전

`built-in chain`과 `RunnableWithMessageHistory`을 이용하여 구현한 코드입니다.

> 08_rag_memory_normal.py

```python
import bs4

from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 1. 블로그 글을 읽고, 청크를 분리하여, 벡터화해서 retriever로 생성
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    )
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()


# 2. 대화기록과 사용자의 질문을 통해, 사용자의 질문에 대화맥락을 포함하여 재구성한다
contextualize_q_system_prompt = (
    """
    Given a chat history and the latest use question which might reference context in the chat history,
    formulate a standalone question which can be understood without the chat history.
    do NOT answer the question, just reformulate it if needed and otherwise return it as is.
    """
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)
#  맥락에 따라 재구성된 질문에 대해 외부 데이터소스로부터 관련문서를 찾아주는 retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# 3. retriever를 문답체인에 합친다.
system_prompt = (
    """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you dont't know the answer, say that you don't know.
    Use three sentences maxium and keep the answer concise.

    {context}
    """
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

# built-in chain 사용: retriever + prompt + LLM
question_answer_chain = create_stuff_documents_chain(llm, prompt) # 모델에 문서목록 전달을 위한 체인
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain) # 문서를 검색한 다음 전달하는 검색 체인


# 4. chat history를 세션으로 관리
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# 5. RAG + history(memory) + Q&A chain 테스트 
qna1 = conversational_rag_chain.invoke(
    {"input": "What is Task Decomposition?"},
    config={
        "configurable": {"session_id": "abc123"}
    },  
)["answer"]

qna2 = conversational_rag_chain.invoke(
    {"input": "What are common ways of doing it?"},
    config={"configurable": {"session_id": "abc123"}},
)["answer"]

print(qna2)
```

위 코드의 결과를 [LangSmith](https://smith.langchain.com/public/eb2231e3-9475-4118-bf38-57911eaab45d/r)를 통해 확인해보겠습니다.

1. 첫 질문인 `What are common ways of doing it?`을 입력받아, 과거 대화 기록을 로드합니다.
2. `retrieval_chain`을 실행합니다
   1. 대화기록과 사용자의 질문을 통해, 사용자의 질문에 대화맥락을 포함하여 재구성합니다. -\> `What are common ways of performing task decomposition?`
   2. 재구성된 질문을 통해 retriever에서 관련문서를 조회하여 반환합니다.
3. `stuff_documents_chain` 을 실행합니다.
4. 질문과 2-2를 통해 반환받은 관련문서를 통해 최종 답을 반환합니다.

### agent 적용

에이전트는 LLM의 추론 기능을 활용하여 실행 중 의사 결정을 내립니다. 에이전트를 적용하면, 검색 프로세스에 대해 일부 재량을 위임할 수 있습니다.

위에서 구현했던 chain보다는 관련 문서를 반환할 때 예측 가능성이 낮지만, 몇 가지 이점이 있습니다.

- 에이전트 적용 시, 위에서 구현했던 것처럼 상황에 맞게 명시적으로 구축할 필요가 없습니다.
- 에어전트 적용 시, 여러 단계의 검색을 실행하지 않아도 됩니다.

위 코드를 agent를 적용해서 다시 작성해보겠습니다.

#### retrieval tool 사용

blog를 retriever로 만든 것을 가지고, `create_retriever_tool`을 만들어봅시다.

```python
from langchain.tools.retriever import create_retriever_tool

tool = create_retriever_tool(
    retriever,
    "blog_post_retriever",
    "Searches and returns excerpts from the Autonomous Agents blog post.",
)
tools = [tool]
```

#### Agent 구성

langgraph에서 제공하는 `create_react_agent`를 통해 agent를 정의해봅시다.

```python
from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(llm, tools)
```

여기까지만 된다면, 아직 해당 RAG Q&A agent는 메모리가 포함되어 있지 않아 맥락을 고려한 대화는 불가능합니다.

[LangSmith](https://smith.langchain.com/public/6f57bca8-3865-4752-b0d1-9a36f9325233/r)를 보면, `What are common ways of doing it?`라는 질문을 했을 때 기존에는 `it`이 `Task Decomposition`을 의미하는 것을 대화기록을 통해 인지했으나, 아직은 인지하지 못하는 것을 확인할 수 있습니다.

#### Memory 추가

영속성 메모리를 추가해볼까요? 기존에는 `RunnableWithMessageHistory`을 이용해서 chat history를 저장, 로딩을 진행했었습니다.

하지만 Agent를 구현하는 LangGraph는 지속성이 내장되어있기 때문에, `RunnableWithMessageHistory`을 사용할 필요가 없습니다.

langgraph내의 `SqliteSaver`를 이용해서 checkpoint를 넘겨주면 됩니다.

> :bulb:Checkpointer ?
>
> - Checkpointer는 에이전트에게 **"메모리"를 제공**하여 상태를 유지할 수 있게 해주는 기능
> - 장기적인 작업을 수행하거나 복잡한 대화를 유지하는 데 필요한 "Memory"를 관리해준다

```python
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")

agent_executor = create_react_agent(llm, tools, checkpointer=memory)
```

위 과정을 적용한 전체 코드입니다.

> 09_rag_memory_agent.py

```python
import bs4
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import HumanMessage
from langchain_core.tools import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 1. 블로그 글을 읽고, 청크를 분리하여, 벡터화해서 retriever로 생성
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    )
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits,
                                    embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# 2. Retrival Tool 정의
tool = create_retriever_tool(
    retriever,
    "blog_post_retriever",
    "Searches and returns excerpts from the Autonomous Agents blog post.",
)
tools = [tool]

# 3. 메모리를 저장하고 있는 checkpointer 정의
memory = SqliteSaver.from_conn_string(":memory:")

# 4. Agent 정의
agent_executor = create_react_agent(llm, tools, checkpointer=memory)

# 5. Agent 실행
query = "What is Task Decomposition?"
print(agent_executor.invoke(
    {"messages": [HumanMessage(content=query)]},
    config={
        "configurable": {"thread_id": "abc123"}
    },
))
query2 = "What are common ways of doing it?"
print(agent_executor.invoke(
    {"messages": [HumanMessage(content=query2)]},
    config={"configurable": {"thread_id": "abc123"}},
))
```

[첫번째 질문에 대한 Langsmith](https://smith.langchain.com/public/78acd800-f911-4abf-a005-38fc4cf1e99d/r)

[두번째 질문에 대한 Langsmith](https://smith.langchain.com/public/52afe6b7-99a7-42f2-aa07-d6c34448ec96/r)

첫 번째 agent는 저희가 만든 retriver tool인 `blog_post_retriever`을 이용해 Task Decomposition에 대한 관련 문서를 조회하고, 이 결과값은 질문과 함께 LLM에 전달하여 답을 받아옵니다.

두 번째 agent는 `What are common ways of doing it?`라는 질문을 통해 우선 첫 번째 agent에서 진행했던 히스토리를 불러옵니다. 이를 통해서 첫 번째 진행했던 과정에서 추출한 정보로 답변할 수 있다는 걸 판단하여 굳이 다시 `blog_post_retriever`를 호출하지 않고, LLM내에서 `What are common ways of doing it?`에 대한 답변을 생성합니다.