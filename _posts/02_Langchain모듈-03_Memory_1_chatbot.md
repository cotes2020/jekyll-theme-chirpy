많은 LLM 애플리케이션은 대화형 인터페이스를 가지고 있습니다. 우리가 흔히 말하는 챗봇과 같은 형식이죠. 대화의 필수 요소 중 하나는 이전에 대화에서 도입된 정보를 참조할 수 있는 능력입니다.

이번 챕터에서는 언어 모델과의 과거 상호작용에 대한 정보를 저장하는 능력인 "Memory"에 대해 배워보도록 하겠습니다. 랭체인에서는 시스템에 메모리를 추가하기 위한 유틸리티를 제공합니다.

`Memory` 모듈에 대한 컨셉에 대해 먼저 알아본 후, 맥락에 따른 대화를 하는 챗봇을 만들어봅시다!

---

## Memory의 컨셉에 대해 알아봅시다

### 1\. 소개

![](https://js.langchain.com/v0.1/assets/images/memory_diagram-0627c68230aa438f9b5419064d63cbbc.png)

메모리 시스템은 `읽기`와 `쓰기` 라는 두 가지 기본 작업을 지원합니다.

모든 체인은 입력에 대해 핵심 로직을 수행합니다. **핵심 로직은 프롬프트 생성, 모델 연동, 출력 파서의 기능을 의미합니다.**

사용자로부터 질문이 입력되면 체인에 바로 전달되어 답변이 생성되기도 하지만, 일부는 메모리를 통해서 입력이 올 수도 있습니다.

체인은 주어진 실행에서 **메모리 시스템과 두 번 상호작용** 합니다.

1. `읽기` 사용자의 초기 입력을 받은 후 핵심 로직을 실행하기 전에, 체인은 **메모리 시스템에서 읽어와서 사용자 입력을 보강**합니다.
2. `쓰기` 핵심 로직을 실행한 후 응답을 반환하기 전에, 체인은 현재 실행의 **입력과 출력을 메모리에 기록**하여 향후 실행에서 참조할 수 있도록 합니다.

### 2\. 메모리 시스템 구축의 핵심 기능

메모리 시스템을 설계할 때는 두 가지의 핵심 고안점이 있습니다.

- 메시지를 저장하는 방법
- 메시지를 검색하는 방법

#### `저장`: 대화 메시지의 목록

모든 대화 메시지의 기록은 메모리의 저장 대상입니다. 랭체인의 메모리 모듈의 주요 부분 중 하나는 이러한 대화 메시지를 저장하기 위한 통합을 제공합니다. 이는 in-memory 목록부터 영구적인 데이터베이스까지 모두 포함합니다.

#### `검색`: 대화 메시지 기반의 데이터 구조와 알고리즘

대화 메시지를 저장하는 것은 비교적 간단합니다. 하지만 메모리에서 대화 메시지를 기반으로 하는 데이터 구조 및 알고리즘을 통해 가장 유용한 메시지를 제공하는지가 중요합니다.

매우 간단한 케이스의 경우 각 실행마다 가장 최근의 메시지를 반환합니다. 혹은 지금까지의 대화의 모든 메시지를 그대로 전달합니다. 좀 더 복잡한 경우에는 과거의 메시지 일부를 간결한 요약으로 반환합니다. 정교하게 추출하는 경우에는 저장된 메시지에서 엔터티를 추출하고 현재 실행에 필요한 엔터티의 대한 정보만 반환하기도 합니다.

각 어플리케이션 별로 최적화된 메모리를 검색하는 방법을 선택해서 사용할 수 있습니다.

> 출처: https://python.langchain.com/v0.1/docs/modules/memory/

---

## 맥락을 고려하는 대화하는 챗봇 만들기

#### `ConversationBufferMemory` 모듈 (기본편!)

이번에는 직접 `Memory`의 `ConversationBufferMemory` 모듈을 이용해서 1)대화 기록을 저장하고, 2)대화 기록을 불러오는 기능을 만들어 보고, 이를 통해 문맥에 맞게 대답하는 챗봇 어플리케이션을 개발해봅시다.

> chat_memory_1.py

```python
import chainlit as cl
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory  #← ConversationBufferMemory 가져오기
from langchain.schema import HumanMessage

chat = ChatOpenAI(
    model="gpt-3.5-turbo"
)

memory = ConversationBufferMemory( #← 메모리 초기화
    return_messages=True #← Chat models에서 Memory 모듈을 사용하기 위해 반드시 설정 ! 
)

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="저는 대화의 맥락을 고려해 답변할 수 있는 채팅봇입니다. 메시지를 입력하세요.").send()

@cl.on_message
async def on_message(message: str):
    memory_message_result = memory.load_memory_variables({}) #← 메모리 내용을 로드

    messages = memory_message_result['history'] #← 메모리 내용에서 메시지만 얻음

    messages.append(HumanMessage(content=message)) #← 사용자의 메시지를 추가

    result = chat( #← Chat models를 사용해 언어 모델을 호출
        messages
    )

    memory.save_context(  #← 메모리에 메시지를 추가
        {
            "input": message,  #← 사용자의 메시지를 input으로 저장
        },
        {
            "output": result.content,  #← AI의 메시지를 output으로 저장
        }
    )
    await cl.Message(content=result.content).send() #← AI의 메시지를 송신
```

위 코드를 실행하기 위해서는 `chainlit` 명령어를 이용해서 `chainlit run chat_memory_1.py`으로 실행합니다.

메모리 모듈의 동작을 위해서는 1)사용자로부터 입력을 받고 2)입력에 따라 저장된 대화 내용에서 메시지를 찾아서 3)질문과 대화 기록을 모델에 전달합니다. 4)모델로부터 받은 응답과 사용자의 입력(질문)을 다시 대화 내용에 저장하는 것을 위에서 보았습니다.

`chat_memory_1.py`도 해당 과정을 수행합니다. 위에서 메모리에서 `검색`하는 방법이 다양하다고 보았는데요, 이번 코드에서는 모든 대화를 그대로 전달하는 방법으로 수행합니다.

메모리 모듈의 주요 동작이 있는 `on_message` 메서드를 살펴봅시다.

- `load_memory_variables`를 통해 저장된 대화 기록을 로드합니다. =\> 2번 과정
- `messages`에 입력된 사용자의 질문과 대화 기록에서의 메시지만 추출한 내용을 저장하여, `messages`를 Chat Model에 전달합니다. =\> 3번 과정
- `save_context`를 통해 사용자가 보낸 메시지와 AI의 메시지를 메모리에 추가합니다. =\> 4번 과정

위 코드를 수행해보면 아래의 흐름을 볼 수 있습니다.

- "계란찜의 재료를 알려줘" 라고 질문
  - `messages`: 프롬프트 입력되는 내용

    ```json
    [HumanMessage(content='계란찜의 재료를 알려줘', additional_kwargs={}, example=False)]
    ```
  - `memory`: model로부터 답변 후 메모리에 저장된 대화 내용

    ```json
     chat_memory
     =ChatMessageHistory(
        messages=[
            HumanMessage(content='계란찜의 재료를 알려줘', ...), HumanMessage(content='계란찜의 재료를 알려줘', ...),
            AIMessage(content='계란찜을 만들기 위해 필요한 재료는 다음과 같습니다:\n\n- 계란\n- 물\n- 소금\n- 설탕\n- 참기름\n- 다진 파 또는 양파 (선택 사항)\n- 간장 (선택 사항)', ...)]) ...
    ```
- 이어서 "영어로 알려줘" 라고 질문
  - `messages`: 프롬프트 입력되는 내용

    ```json
    HumanMessage(content='계란찜의 재료를 알려줘', additional_kwargs={}, example=False), 
    HumanMessage(content='계란찜의 재료를 알려줘', additional_kwargs={}, example=False), 
    AIMessage(content=' 
    계란찜을 만들기 위해 필요한 재료는 다음과 같습니다:\n\n- 계란\n- 물\n- 소금\n- 설탕\n- 참기름\n- 다진 파 또는 양파 (선택 사항)\n- 간장 (선택 사항)', additional_kwargs={}, example=False), 
    HumanMessage(content='영어로 알려줘', additional_kwargs={}, example=False)]
    ```
  - `memory`: model로부터 답변 후 메모리에 저장된 대화 내용

    ```json
    chat_memory=ChatMessageHistory(
        messages=[
            HumanMessage(content='계란찜의 재료를 알려줘', additional_kwargs={}, example=False), 
            HumanMessage(content='계란찜의 재료를 알려줘', additional_kwargs={}, example=False), 
            AIMessage(content='계란찜을 만들기 위해 필요한 재료는 다음과 같습니다:\n\n- 계란\n- 물\n- 소금\n- 설탕\n- 참기름\n- 다진 파 또는 양파 (선택 사항)\n- 간장 (선택 사항)', additional_kwargs={}, example=False), 
            HumanMessage(content='영어로 알려줘', additional_kwargs={}, example=False), 
            HumanMessage(content='영어로 알려줘', additional_kwargs={}, example=False), 
            AIMessage(content='Here are the ingredients for "Gyeranjjim" (Korean Steamed Eggs):\n\n- Eggs\n- Water\n- Salt\n- Sugar\n- Sesame oil\n- Minced green onions or onions (optional)\n- Soy sauce (optional)', additional_kwargs={}, example=False)]) ...
    ```

이처럼 memory에 질문은 `HumanMessage`로, 모델의 답변은 `AIMessage`로 계속 쌓여서 저장이 됩니다. 저장된 내용은 다음 질문 시 앞에 추가되어 새로 입력된 질문과 함께 모델에게 프롬프트로 전달됩니다.

#### `ConversationChain` 모듈 (응용편!)

위 코드에서는 Memory에 저장, Memory에서 과거 메시지 가져오기, 언어 모델 호출을 각각 코드로 구현하였으나, `ConversationChain` 모듈 사용 시 위 과정을 하나의 모듈로 수행할 수 있습니다.

> chat_model_2.py

```python
import chainlit as cl
from langchain.chains import ConversationChain  #← ConversationChain을 가져오기
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

chat = ChatOpenAI(
    model="gpt-3.5-turbo"
)

memory = ConversationBufferMemory( 
    return_messages=True
)

chain = ConversationChain( #← ConversationChain을 초기화
    memory=memory,
    llm=chat,
)

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="저는 대화의 맥락을 고려해 답변할 수 있는 채팅봇입니다. 메시지를 입력하세요.").send()

@cl.on_message
async def on_message(message: str):

    result = chain( #← ConversationChain을 사용해 언어 모델을 호출
        message #← 사용자 메시지를 인수로 지정
    )

    await cl.Message(content=result["response"]).send()
```

`ConversationChain`을 초기화하여 `ConversationBufferMemory`으로 초기화한 memory와 llm(모델)을 할당해줍니다. 이 체인 객체에 새로 입력받은 message를 파라미터로 입력하여 호출해주면 됩니다.

`ConversationChain`을 사용해서 아래의 과정이 하나의 함수 호출로 처리된 것을 확인할 수 있습니다.

1. 메모리에서 과거 메시지 검색
2. 새로운 메시지 추가
3. 이 메시지를 언어 모델에 전달해 새로운 응답 얻기
4. 새로운 응답을 메모리에 저장

`ConversationChain`을 이용하면 보다 편리한 개발이 가능합니다 :) !

---

### :pencil2:️ Wrap up!

이번 글에서는 랭체인에서 대화 기록을 저장하고, 저장된 내용을 호출하는 방법을 통해 맥락적 대화가 가능한 기능을 구현하기 위한 `Memory` 모듈에 대해 간단히 살펴보았습니다.

`Memory` 모듈에도 사람의 기억처럼 단기기억, 장기기억이 존재합니다! 이번에는 어플리케이션이 종료되면 사라지는 단기기억에 가까운 서비스를 구현해보았는데요, 다음 글에서는 대화 내용을 데이터베이스에 영속적으로 관리하여 장기기억으로 관리할 수 있는 방법에 대해 알아봅시다!