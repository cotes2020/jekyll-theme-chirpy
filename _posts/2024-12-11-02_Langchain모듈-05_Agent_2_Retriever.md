---
title: [Langchain Study | 3-2. Agents 2 - Agent + Retrievers]
categories: [AI, Langchain]
tags: [Langchain, AI, AI Application, Agent]		
---



언어 모델이 모르는 정보에 대해 대답하게 하는 기법인 RAG도 Agent에 적용해볼 수 있습니다.

바로 Retrievers를 Tool로 변환하여 간단하게 적용해줄 수 있습니다.

이번 글에서는 Retreivers를 Tool로 변환하여 Agent에서 사용해보도록 하겠습니다!

---

## WikipediaRetriever를 Agent의 Tool로 적용해보기

Langchain에서는 위키피디아 정보를 정보원으로 활용할 수 있는 retriver인 `WikipediaRetriever`를 제공하고 있습니다. 이 retriever를 이용해서 위키백과에서 입력된 토픽에 대해 찾아볼 수 있습니다.

우리는 agent과의 결합을 통해 위키피디아에 키워드에 대해 물어보고, 답변을 한국어로 변환해서 로컬에 텍스트파일로 저장해주는 코드를 구현해봅시다.

그러면 우선 위 과정을 위해서는 두 가지 툴이 필요합니다.

- 위키피디아에 키워드 검색

- 파일 저장

여기서 우리는 `위키피디아에 키워드 검색` 툴을 `WikipediaRetriever`를 가지고 구현해볼 예정입니다.

> agent_rag.py

```python
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.agents.agent_toolkits import create_retriever_tool  #← create_retriever_tool을 가져오기
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import WikipediaRetriever #←WikipediaRetriever를 가져오기
from langchain.tools import WriteFileTool

chat = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo"
)

tools = []

tools.append(WriteFileTool(
    root_dir="./"
))

retriever = WikipediaRetriever( #←WikipediaRetriever를 초기화
    lang="ko", #←언어를 한국어로 설정
    doc_content_chars_max=500,  #←글의 최대 글자 수를 500자로 설정
    top_k_results=1 #←검색 결과 중 상위 1건을 가져옴
)

tools.append(
    create_retriever_tool(  #←Retrievers를 사용하는 Tool을 생성
        name="WikipediaRetriever",  #←Tool 이름
        description="입력된 단어에 대한 Wikipedia 기사를 검색할 수 있다",  #←Tool 설명
        retriever=retriever,  #←Retrievers를 지정
    )
)

agent = initialize_agent(
    tools,
    chat,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run("판나코타에 대해 Wikipedia에서 찾아보고 그 개요를 한국어로 result.txt 파일에 저장하세요.")

print(f"실행 결과: {result}")
```

`create_retriever_tool`을 이용해서 Retrievers를 사용하는 Tool을 생성합니다. 기존에 Tool을 생성했던 것과 마찬가지로 `name`과 `description`을 기재합니다. 직전에 직접 툴을 생성할 때는 `func` 파라미터를 통해 처리할 액션을 함수로 지정해주었는데요, 이번에는 `retriever`에 Tool화할 Retrievers를 지정해주도록 합니다. 우리는 위에서 선언한 `WikipediaRetriever`를 지정해주었습니다.

`initialize_agent`를 통해 tools와 chat을 지정해주고, 여러가지 툴을 이용하기 때문에 STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION 방식을 선언해줍니다.

코드를 수행하면 `WikipediaRetriever`를 통해 위키피디아에서 판나코다에 대해 검색하고, 검색한 결과를 `result.txt`로 저장해줍니다.

---

### ✏️ Wrap up!

이번 예제에서는 Retriever를 Tool로 변환하여 Agent에서 활용할 수 있는 예제를 만들어보았습니다.

이번엔 이미 존재하는 retriever인 `WikipediaRetriever`을 통해 구현해보았지만, 저희가 직접 임베딩하여 저장해둔 벡터 DB을 retriever로 변환하여 같은 방법으로 구현해볼 수 있습니다.

[PDF기반 챗봇만들기 (실습)](https://git.bwg.co.kr/gitlab/study/langchain/-/wikis/02_Langchain%EB%AA%A8%EB%93%88-02_Retrieval_3_PDF%EA%B8%B0%EB%B0%98-%EC%B1%97%EB%B4%87%EB%A7%8C%EB%93%A4%EA%B8%B0-(%EC%8B%A4%EC%8A%B5))에서 생성해둔 vector databse retriever를 통해 비행기자동차의 최고 속도에 대한 정보를 영어로 번역해서 text 파일로 로컬에 저장해주는 코드를 한 번 구현해보시는 것도 재밌는 경험이 되실 거예요 :)

힌트를 드리자면 벡터 DB을 `as_retriever()`를 통해 retriever로 변환할 수 있답니다!

다음은 Memory 모듈을 Agent에 결합하여, 문맥에 맞게 답변하는 에이전트를 만들어보겠습니다!




