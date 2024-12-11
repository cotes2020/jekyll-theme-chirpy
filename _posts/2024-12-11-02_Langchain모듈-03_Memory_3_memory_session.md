---
title: [Langchain Study | 3-2. Memory 3 - RunnableWithMessageHistory, Memory Session]
categories: [AI, Langchain]
tags: [Langchain, AI, AI Application, Memory]		
---


이번 챕터에서는 `Langchain verion2`에서 제공하는 모듈을 이용해보겠습니다!

[가상환경 구축](https://git.bwg.co.kr/gitlab/study/langchain/-/wikis/02_Langchain%EB%AA%A8%EB%93%88-00\_%EA%B0%80%EC%83%81%ED%99%98%EA%B2%BD-%EA%B5%AC%EC%B6%95)을 참고하여 라이브러리 버전을 Langchain version2로 맞춰주세요\~

이번 챕터에서는 대화기록을 `Memory`에 저장했었는데요, 이것을 이용자 별, 즉 세션 별로 구분해서 대화기록을 관리하는 챗봇을 만들어보겠습니다.

Langchain의 `RunnableWithMessageHistory`를 이용해보겠습니다. [`RunnableWithMessageHistory`](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html#langchain_core.runnables.history.RunnableWithMessageHistory) class는 각 세션을 `session_id`로 구분하여 관리해줍니다!

---

## Step 1.`RunnableWithMessageHistory` 의 구조

챗봇 구축 시에는 대화 상태의 세션을 분리하여 관리하는 것이 중요합니다.

`RunnableWithMessageHistory`클래스를 이용하면 기존 메시지를 로드하여 작업을 수행하고, 다시 이 과정을 저장하는 기록을 관리해줍니다.

Runnable(수행 작업) 전에 대화의 이전 메시지를 로드하고, Runnable을 호출한 후 생성된 응답을 다시 메시지로 저장해주는 것이죠.

이때 `session_id`를 통해 각 대화를 저장하는 것을 통해 여러 대화를 가능하게 합니다.

![image-20240704152905767]({{"/assets/img/posts/image-20240704152905767.png"  | relative_url }})

## Step 2. 구축 방법

우리는 `RunnableWithMessageHistory`클래스를 사용하기 위해 아래 두가지 질문을 가지고 살펴보도록 하겠습니다.

- `messages`를 어떻게 저장하고, 로드하는가?
- 감싸진 `underlying Runnable`이 무엇이고, 그것의 입력/출력값이 무엇인가?

#### message 저장/로드 방법

`RunnableWithMessageHistory`을 구축하기 위해서는 `get_session_history` 함수가 필요합니다.

`get_session_history` 는 `session_id`을 입력받아서 `BaseChatMessageHistory` 객체를 반환해줍니다.

> \[!TIP\]
>
> - **`session_id`**: 세션(대화) 스레드의 식별값, 동시에 여러 대화(스레드)를 같은 체인 내에서 유지할 수 있게 해줌
> - **`BaseChatMessageHistory`**: 대화기록의 로딩과 저장을 관리해주는 클래스. `RunnableWithMessageHistory`의 구성요소로 사용됨

`get_session_history` 함수를 만들어볼까요? BaseChatMessageHistory로는 간단한 구현을 위해 `SQLChatMessageHistory`을 사용하였습니다.

```python
from langchain_community.chat_message_histories import SQLChatMessageHistory

def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///memory.db")
```

- `sqlite`를 통해서 파일 db로 `memory.db`라는 db 저장소에 대화내용을 저장합니다.
- 입력값으로 session_id를 받아서, session_id 별로 대화내용을 별도로 관리해줍니다.

#### 어떤 `runnable`을 사용하는지

> \[!NOTE\]
>
> **`Runnable`이란?**
>
> - [Runnable interface](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable)
> - Langchain에서 invoke, batch, stream, transform&compose할 수 있는 **작업 단위**
> - Lagnchain v2에서 제공하는 LCEL을 통해 `Runnable`(작업단위)를 체인으로 구성할 수 있습니다.
>   - 순차적으로 수행하는 `RunnableSequence` 방식
>   - 병렬적으로 수행하는 `RunnableParallel` 방식
> - Langchain에서 `runnables`를 통해 데이터 처리를 효율화하고, 복잡한 워크플로우를 구성할 수 있습니다.
> - 예를 들어, 프롬프트 템플릿으로 작성, LLM 수행, 결과값 반환과 같은 3개의 작업을 하나의 체인으로 구성해 `Runnables`로 만들 수 있습니다.

`RunnableWithMessageHistory`은 `runnable`을 감싸고 있습니다.

이때 runnable의 입력/출력 타입은 [지정된 타입](https://python.langchain.com/v0.2/docs/how_to/message_history/#what-is-the-runnable-you-are-trying-to-wrap)인 경우 사용이 가능합니다. 저희는 **"입력:`dictonary` - 출력:`AI 메시지`"** 케이스를 살펴보겠습니다.

> 이때, 입력이 되는 메시지는 langchain 스키마에 존재하는 모든 메시지 타입이 가능합니다. (Model I/O에서 보았던 [Message Types](https://git.bwg.co.kr/gitlab/study/langchain/-/wikis/02_Langchain%EB%AA%A8%EB%93%88-01_Model-IO#message-types) 참고)

위에서 만들었던 `get_session_history` 함수를 적용하고,

`runnable`은  Prompt Template + LLM으로 구성된 chain을 이용해서 `RunnableWithMessageHistory`을 구현해봅시다.

> 05_msg_history_RunnableWithMessageHistory.py

```python
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("human", "{input_message}"),
])

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# chain으로 runnable 구성
chain = prompt | llm


def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///memory.db")


runnable_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
)

# 첫번째 메시지 전송
runnable_with_history.invoke(
    {"input_message": "hi - im bob!"},
    config={"configurable": {"session_id": "1"}},
)

# 두번째 메시지 전송
result = runnable_with_history.invoke(
    {"input_message": "whats my name?"},
    config={"configurable": {"session_id": "1"}},
)

print(result.content)
```

- `runnable` 구성: LCEL을 통해 chain으로 runnable을 구성합니다.
  - prompt: chat prompt template을 이용해서 human스키마로 입력 메시지 생성
  - llm: OpenAI 모델 연동
- 위에서 만든 `get_session_history`와 `runnable`을 이용해 `RunnableWithMessageHistory`을 초기화합니다.
  - 위 코드에서 `RunnableWithMessageHistory`가 감싸고 있는 `runnable`은 `chain(prompt+llm)` 입니다.
- `RunnableWithMessageHistory`를 `invoke`하여 실행합니다.
  - input: prompt에 입력될 말을 넣어줍니다.
  - config: session_id를 지정해줍니다.

위 코드를 수행하면 두번째 질문에 대한 답으로 아래와 같이 생성됩니다. 첫번째 질문을 통해 제가 bob이라고 말했던 것을 기억하고 있는 것이죠.

```output
Your name is Bob.
```

만약 session_2에서 제 이름을 다시 물어보면 기억하지 못합니다.

```python
runnable_with_history.invoke(
    {"input_message": "whats my name?"},
    config={"configurable": {"session_id": "2"}},
)
```

```output
I'm sorry, but I don't have access to your personal information, including your name. If you would like to share your name with me, feel free to do so!
```

### (deep!) prompt에 여러 개의 variable 사용해보기

- 위 예제로 prompt 템플릿을 입력으로 받기 때문에 input: dictionary - output: message 구조였지만, prompt 변수로 여러 개를 지정해볼 수도 있습니다.

  <details>
  <summary>소스코드</summary>

  ```python
  from langchain_community.chat_message_histories import SQLChatMessageHistory
  from langchain_openai import ChatOpenAI
  from langchain_core.runnables.history import RunnableWithMessageHistory
  from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
  
  prompt = ChatPromptTemplate.from_messages(
  [
  (
  "system",
  "You're an assistant who always speaks in {language}. Respond in 20 words or fewer",
  ),
  MessagesPlaceholder(variable_name="history"),
  ("human", "{input}"),
  ]
  )
  
  llm = ChatOpenAI(model="gpt-4o")
  
  # chain으로 runnable 구성
  chain = prompt | llm
  
  def get_session_history(session_id):
  return SQLChatMessageHistory(session_id, "sqlite:///memory.db")
  
  runnable_with_history = RunnableWithMessageHistory(
  chain,
  get_session_history,
  input_messages_key="input",
  # 대화 히스토리가 하나의 Human Message안에 포함되어 프롬트로 전달되지 않고,
  # Human/AI/Human/.. 순으로 content만 추출되어 전달됨
  history_messages_key="history"
  )
  
  # 첫번째 메시지 전송
  runnable_with_history.invoke(
  {"language":"Korean","input": "hi - im bob!"},
  config={"configurable": {"session_id": "1"}},
  )
  
  # 두번째 메시지 전송
  session1 = runnable_with_history.invoke(
  {"language":"Korean", "input": "whats my name?"},
  config={"configurable": {"session_id": "1"}},
  )
  
  # 다른 세션에 메시지 전송
  session2 = runnable_with_history.invoke(
  {"language": "italian", "input": "whats my name?"},
  config={"configurable": {"session_id": "2"}},
  )
  
  print(session1.content)
  print(session2.content)
  ```

  </details>

  - **prompt**
    - ChatPromptTemplate으로 구성
    - system message: {language} 지정
    - MessagePlaceholder를 통해 대화 기록 전달 - {history}
    - human message: {input}
  - **RunnableWithMessageHistory 초기화**
    - chain지정(prompt+llm)
    - get_session_history 함수 지정
    - input_message_key \<- `{input}` 할당 : 가장 최근에 입력받은 메시지를 할당하는 키 (사용자의 입력 메시지를 식별)
    - history_messages_key \<- `{history}` 할당: 대화기록을 할당하는 키 (대화 이력에 접근하여 이를 프롬프트에 포함)

  위 코드를 수행하면 아래와 같이 결과가 나옵니다.

  ```output
  당신의 이름은 Bob입니다.
  Mi dispiace, non conosco il tuo nome. Puoi dirmelo tu?
  ```

  session1은 한국어로 답하고, Bob이라는 이름을 기억하지만,

  session2는 이탈리어로 답하고, 이름에 대한 대화를 한 적이 없기 때문에 이름을 기억하지 못합니다.

## Step 3. application에 적용해보기

앞 챕터에서 만들었던 [맥락 고려하는 챗봇](https://git.bwg.co.kr/gitlab/study/langchain/-/wikis/02_Langchain%EB%AA%A8%EB%93%88-03_Memory_1_chatbot#%EB%A7%A5%EB%9D%BD%EC%9D%84-%EA%B3%A0%EB%A0%A4%ED%95%98%EB%8A%94-%EB%8C%80%ED%99%94%ED%95%98%EB%8A%94-%EC%B1%97%EB%B4%87-%EB%A7%8C%EB%93%A4%EA%B8%B0)에 `RunnableWithMessageHistory`를 적용해봅시다!

원래는 하나의 세션에서 수행되는 일회성 챗봇이었습니다. `RunnableWithMessageHistory`을 적용함으로써 두 가지 기능이 추가됩니다.

- 세션 별로 구분되는 대화 기록
- 어플리케이션이 종료된 후에도 대화 기록 유지

구현할 어플리케이션의 work flow입니다.

- 대화 시작 시에 사용자로부터 `세션ID`를 입력받습니다.
- 과거에 대화했던 기록이 있다면 기존 대화 기록을 채팅창에 보여줍니다.
- 해당 대화기록을 유지하며 대화를 진행합니다.

> 05_chat_memory_5.py

````python
import chainlit as cl
from langchain.chat_models import ChatOpenAI
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory

# RunnableWithMessageHistory을 이용한 세션 + Memory + 데이터 영속화 코드

# Chat model과 Prompt Template 초기화
chat = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친근하게 말하는 챗봇입니다."),
    ("placeholder", "{history}"),
    ("human", "{input}"),
])

# 대화 chain 생성
chain = prompt | chat

# DB에서 세션에 따라 대화기록 반환받는 함수 정의
def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///memory.db")

# RunnableWithMessageHistory 초기화
runnable_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 대화 시작 함수 정의
@cl.on_chat_start
async def on_chat_start():

    # session_id 입력받기
    session_id = None
    while not session_id:
        res = await cl.AskUserMessage(
            content="저는 대화의 맥락을 고려해 답변할 수 있는 채팅봇입니다. 세션ID를 입력하세요.",
            timeout=600).send()
        if res:
            session_id = res['output']
    # 입력받은 세션ID를 chainlit session에 저장
    cl.user_session.set("session_id", session_id)

    # 입력된 세션의 메시지 히스토리 불러오기
    messages = get_session_history(session_id).get_messages()

    # 메시지 대화 기록 하나의 문자열로 formatting
    messages_history_string = ""
    for message in messages:
        if isinstance(message, HumanMessage):
            messages_history_string += f"\n😀YOU : {message.content}"
        else:
            messages_history_string += f"\n🤖BOT : {message.content}"

    # 대화 기록 화면에 채팅 메시지로 출력
    if messages_history_string != "":
        await cl.Message(content=f"```{messages_history_string}").send()
        return await cl.Message(content="기존에 나눴던 대화 기록입니다. 이어서 대화를 시작하세요!").send()
    else:
        return await cl.Message(content="새로운 대화를 시작하세요!").send()

# 사용자로부터 메시지 받았을 때 수행 함수 정의
@cl.on_message
async def on_message(message: str):
    # 진행 중인 session 획득
    session_id = cl.user_session.get("session_id")

    # 입력받은 message로 (+대화기록) chain runnable 수행 후, session_id에 따라 DB에 대화기록 저장
    result = runnable_with_history.invoke(
        {"input": message.content},
        config={"configurable": {"session_id": session_id}},
    )

    # LLM 답변 채팅 메시지로 출력
    return await cl.Message(content=result.content).send()
````

```
chainlit run 05_chat_memory_5.py
```

위 chainlit 실행 명령어를 통해 어플리케이션을 실행시켜봅니다.

#### 어플리케이션 수행 결과

- 세션 입력

  ![image-20240708101433227]({{"/assets/img/posts/image-20240708101433227.png"  | relative_url }})
  - 어플리케이션을 수행하면 세션ID를 입력하라는 메시지가 먼저 수행됩니다.
  - `밀구`라는 세션으로 대화를 시작합니다.
  - 기존에 대화한 적이 없는 세션이기 때문에 `새로운 대화를 시작하세요!`라고 나옵니다.
- 대화

  ![image-20240708101554997]({{"/assets/img/posts/image-20240708101554997.png"  | relative_url }})
  - 위와 같이 `밀구` 세션에서 대화를 이어갑니다.
- 기존 대화 불러오기

  ![image-20240708101650163]({{"/assets/img/posts/image-20240708101650163.png"  | relative_url }})
  - 대화 종료 후 다시 어플리케이션을 실행합니다.
  - `밀구`라고 세션을 다시 입력합니다.
  - 그 전에 대화 기록을 보여준 후 다시 대화를 이어갈 수 있습니다.

#### `RunnableWithMessageHistory` 진행 과정 보기

위 대화를 [LangSmith](https://smith.langchain.com/public/5502dabb-2a85-4fc7-acac-ac3160093139/r) 의 기록을 통해 어떤 과정으로 진행되었는지 자세히 볼까요?

- **입력 질문 history에 저장**
  - 우선 `insert_history`를 통해 입력된 질문인 `오늘 비가 와서 집에 가고싶어`라는 문장을 기존 대화 기록 history에 저장합니다.
- **기록 대화 history 로드**
  - `load_history`를 통해 기존 대화 기록 history를 불러옵니다.
- **(기존 대화 history + 입력 질문) =\> LLM에 프롬프트 전달**
- **LLM 답변 history에 저장**
  - `RunnableWithMessageHistory`은 `아, 비 오는 날엔 집에서 따뜻한 차 한 잔 마시면서 쉬고 싶어지지. 혹시 집에 가면 뭐하고 싶어?`라는 AI의 답변을 다시 history memory에 저장합니다.
- **LLM 답변 반환**
  - Memory에 가장 최근의 대화 문답까지 저장한 후, 최종적으로 LLM으로부터 생성된 답변을 반환합니다.

---

LangChain의 버전 2에서 제공하는 `RunnableWithMessageHistory`의 원리와 사용법에 대해 알아보았습니다.

`RunnableWithMessageHistory`을 사용하면 메모리에 대화를 저장, 기존 대화를 로드해주는 과정을 별도의 코딩없이 실행할 수 있어 매우 편리하게 코딩이 가능했습니다!