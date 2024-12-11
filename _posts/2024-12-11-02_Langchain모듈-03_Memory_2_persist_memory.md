---
title: [Langchain Study | 3-2. Memory 2 - 영속적 메모리 구현하기 (Persist Memory)]
categories: [AI, Langchain]
tags: [Langchain, AI, AI Application, Memory]		
---


이전 챕터에서 대화 세션 내에서 대화 기록을 저장하는 `Memory` 모듈을 사용해보았습니다. 

어플리케이션 프로그램이 종료되어도 기존의 대화 내용을 저장하는 방법이 있는데요, 대화 내용을 데이터베이스에 저장하여 영속화하는 방법입니다. 이번 챕터에서는 앱 종료 후에도 소통을 재개할 수 있는 어플리케이션을 개발해보도록 하겠습니다!

---

### Docker로 Redis 서버 구동

기존에 연결할 Redis 배포가 없는 경우, 로컬 Redis Stack 서버를 시작합니다.

다음은 Docker 로 Redis 서버를 구동하는 명령어입니다.

```bash
docker run -d -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

`REDIS_URL` 변수에 Redis 데이터베이스의 연결 URL을 할당합니다.

- URL은 `"redis://localhost:6379/0"`로 설정되어 있습니다.

```python
# Redis 서버의 URL을 지정합니다.
REDIS_URL = "redis://localhost:6379/0"
```

### 라이브러리 다운로드

이번 실습을 수행하기 위해 추가적으로 설치해야 할 라이브러리입니다.

1. 가상환경 활성화
   
   ```shell
   .\{가상환경폴더명}\Scripts\activate
   ```

2. 라이브러리 설치
   
   ```shell
   pip install redis
   ```

### 코드 개발

> chat_memory_redis.py

```python
import chainlit as cl
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import RedisChatMessageHistory  #← RedisChatMessageHistory를 추가
from langchain.memory import ConversationBufferMemory

chat = ChatOpenAI(
    model="gpt-3.5-turbo"
)

# RedisChatMessageHistory 초기화
history = RedisChatMessageHistory(
    session_id="chat_history",
    # Redis DB가 떠있는 Docker의 주소
    url="redis://{host}:6379/0"
)

memory = ConversationBufferMemory(
    return_messages=True,
    chat_memory=history,  #← 채팅 기록을 지정
)

chain = ConversationChain(
    memory=memory,
    llm=chat,
)

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="저는 대화의 맥락을 고려해 답변할 수 있는 채팅봇입니다. 메시지를 입력하세요.").send()

@cl.on_message
async def on_message(message: str):

    result = chain(message)

    await cl.Message(content=result["response"]).send()
```

저는 linux 서버에 docker를 띄웠기 때문에 해당 주소로 경로를 설정했습니다. 각자의 환경에 맞춰 경로를 작성해주세요!

`RedisChatMessageHistory`를 호출하여 Redis에 대화 기록을 처리하도록 합니다.

- `session_id`는 임의의 문자열을 지정하며, 현재 대화 세션의 식별자 역할을 합니다. 해당 코드의 어플리케이션 수행 시에는 `"chat_history"`라고 고정값이 들어가기 때문에 항상 같은 대화 기록을 사용합니다.

- `url`은 저장 대상 데이터베이스의 경로를 기재합니다.

`ConversationBufferMemory`에 `chat_memory=history`를 추가하여 채팅 기록을 지정하여, 대화 내역이 레디스에 저장되도록 합니다.



위 코드를 실행하면 기존의 코드와 똑같이 수행되는 것처럼 보입니다. 하지만 어플리케이션을 종료했다가 다시 실행해도, 이전의 대화 기록을 유지하고 있습니다. 화면으로 실행 동작을 보여드리겠습니다.

처음 실행 후 다람쥐를 다랑어라고 부르겠다고 입력합니다. 

![]({{"/assets/img/posts/2024-06-04-16-38-02-image.png" | relative_url }})

어플리케이션 종료 후 다시 실행하여 새로운 대화 세션에서 다람쥐를 뭐라고 부르는지 질문하니 대화 내용을 기억하고 있다가 다랑어라고 답변합니다.

![]({{"/assets/img/posts/2024-06-04-16-38-15-image.png" | relative_url }})

위와 같이 `RedisChatMessageHistory`와 `ConversationBufferMemory`를 결합하여 대화 내역을 DB에 저장하고, 어플리케이션 종료 후에도 내역을 유지할 수 있는 방법에 대해 알아보았습니다.

---

### ✏️ Wrap up!

이번 예시에서는 Redis를 사용하였지만, DB의 대상은 다른 종류를 선택하셔도 됩니다! 보통의 어플리케이션에서는 단발성으로 대화가 끝나는 서비스보다는 한 사용자의 대화 기록을 유지해주는 경우가 많아 대화 내용을 영속화하는 방법은 매우 유용하게 사용됩니다.

ChatGPT의 모습을 생각해보면 어플리케이션 종료 전 대화가 유지되기도 하지만, 새로운 세션으로 대화를 시작하기도 합니다. 즉, 하나의 어플리케이션에서 여러 개 세션의 대화가 진행되기도 합니다! 여러 개의 대화 기록을 가질 수 있는 챗봇도 구현할 수 있는데요, [해당 코드](https://github.com/wikibook/langchain/blob/master/04_memory/chat_memory_4.py)에서 이를 구현하고 있습니다. 관심이 있으신 분들은 한 번 따라서 진행해보시는 걸 추천드립니다!


지금까지 저희가 랭체인의 여러 모듈들을 다뤄보았는데요, 지금까지의 코드에서도 살짝 살짝 나왔었는데요~ 
이러한 모듈들의 연동을 쉽게 해주는 `Chain`에 대해서 다음 글에서 다뤄보도록 하겠습니다 :) 
