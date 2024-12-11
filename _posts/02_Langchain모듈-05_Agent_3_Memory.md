`Memory` 모듈과 `Agents` 모듈을 조합하여 문맥에 따라 응답할 수 있는 에이전트를 만들어봅시다!

---

## 대화기록을 보관하는 에이전트 생성하기

Memory 모듈과의 결합을 통해 대화형 상호작용을 하는 에이전트를 만들어보겠습니다. 

앞 예제와 마찬가지로 위키피디아 retriever를 이용하여 질문을 하고, 그 질문에 이어 다음 질문을 했을 때 이전 질문을 기억한 답변을 해주도록하는 코드입니다.

> agent_memory.py

```python
from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory  #←ConversationBufferMemory 가져오기
from langchain.retrievers import WikipediaRetriever

chat = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo"
)

tools = []

# WriteFileTool을 제거

retriever = WikipediaRetriever(
    lang="ko",
    doc_content_chars_max=500,
    top_k_results=1
)

tools.append(
    create_retriever_tool(  #←Retrievers를 사용하는 Tool을 생성
        name="WikipediaRetriever",  #←Tool 이름
        description="받은 단어에 대한 Wikipedia 기사를 검색할 수 있다",  #←Tool 설명
        retriever=retriever,  #←Retrievers를 지정
    )
)

memory = ConversationBufferMemory(  #←ConversationBufferMemory를 초기화
    memory_key="chat_history",  #←메모리 키를 설정
    return_messages=True  #←메시지를 반환하도록 설정
)

agent = initialize_agent(
    tools,
    chat,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,  #←Agent의 유형을 대화형으로 변경
    memory=memory,  #←Memory를 지정
    verbose=True
)

result1 = agent.run("판나코타에 대해 Wikipedia에서 찾아보고 그 개요를 한국어로 알려주세요.")
print(f"1차 실행 결과: {result1}")
result2 = agent.run("역사에 대해서도 알려주세요.")
print(f"2차 실행 결과: {result2}")
```

Agents의 모듈을 Memory 모듈과 함께 하용하기 위해서는 `AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION`을 사용해야 합니다. 이 방식은 여러 개의 입력을 가진 Tool을 사용할 수 없기 때문에 이번 예제에서는 파일로 저장하는 Tool은 제외하였습니다.

`ConversationBufferMemory`를 이용해 memory를 초기화 해주고, 해당 memory를 `initialize_agent`에서 agent를 초기화할 때 항목으로 할당합니다.





코드를 수행하면 아래의 결과를 받을 수 있습니다.

```log
> Entering new AgentExecutor chain...
"""json
{
    "action": "WikipediaRetriever",
    "action_input": "판나코타"
}
"""
Observation: [Document(page_content="판나코타(이탈리아어: Panna cotta)는 설탕을 넣은 크림을 젤라틴과 함께 굳히고 모양을 만든 이탈리아의 후식이다. 이탈리아어로 '요리한 크림'이라는 뜻을 가진다. --중략--이탈리아어로 '요리한 크림'이라는 뜻을 가진다. 커피와 바닐라 등으로 크림에 향을 낼 수 있다.", 'source': 'https://ko.wikipedia.org/wiki/%ED%8C%90%EB%82%98_%EC%BD%94%ED%83%80'})]
Thought:"""json
{
    "action": "Final Answer",
    "action_input": "판나코타(이탈리아어: Panna cotta)는 설탕을 넣은 크림을 젤라틴과 함께 굳히고 모양을 만든 이탈리아의 후식이다. 이탈리아어로 '요리한 크림'이라는 뜻을 가진다. 커피와 바닐라 등으로 크림에 향을 낼 수 있다."
}
"""

> Finished chain.
1차 실행 결과: 판나코타(이탈리아어: Panna cotta)는 설탕을 넣은 크림을 젤라틴과 함께 굳히고 모양을 만든 이탈리아의 후식이다. 이탈리아어로 '요리한 크림'이라는 뜻을 가진다. 커피와 바닐라 등으로 크림에 향을 낼 수 있다.


> Entering new AgentExecutor chain...
"""json
{
    "action": "WikipediaRetriever",
    "action_input": "판나코타 역사"
}
"""
Observation: [Document(page_content="판나코타(이탈리아어: Panna cotta)는 설탕을 넣은 크림을 젤라틴과 함께 굳히고 모양을 만든 이탈리아의 후식이다. 이탈리아어로 '요리한 크림'이라는 뜻을 가진다. --중략--이탈리아어로 '요리한 크림'이라는 뜻을 가진다. 커피와 바닐라 등으로 크림에 향을 낼 수 있다.", 'source': 'https://ko.wikipedia.org/wiki/%ED%8C%90%EB%82%98_%EC%BD%94%ED%83%80'})]
Thought:"""json
{
    "action": "Final Answer",
    "action_input": "판나코타는 이탈리아의 후식으로, 설탕을 넣은 크림을 젤라틴과 함께 굳히고 모양을 만든다. 이탈리아어로 '요리한 크림'이라는 뜻을 가지며, 커피와 바닐라 등으로 향을 낼 수 있다. 역사적으로는 20세기 중반까지 이탈리아 요리책에 언급되지 않았으나, 피에몬테주의 전통 디저트로 자주 거론되며, 한 헝가리 여성이 19세기 초 랑게에서 이를 발명했다는 이야기도 있다."
}
"""

> Finished chain.
2차 실행 결과: 판나코타는 이탈리아의 후식으로, 설탕을 넣은 크림을 젤라틴과 함께 굳히고 모양을 만든다. 이탈리아어로 '요리한 크림'이라는 뜻을 가지며, 커피와 바닐라 등으로 향을 낼 수 있다. 역사적으로는 20세기 중반까지 이탈리아 요리책에 언급되지 않았으나, 피에몬테주의 전통 디저트로 자주 거론되며, 한 헝가리 여성이 19세기 초 랑게에서 이를 발명했다는 이야기도 있다.
```

첫번째 질문인 `판나코타에 대해 Wikipedia에서 찾아보고 그 개요를 한국어로 알려주세요.`에 대해 답변을 줍니다. 단순히 판나코타에 대한 개요 설명입니다.

두번째 질문은 `역사에 대해서도 알려주세요.`이며 무엇의 역사인지에 대한 설명은 넣지 않았습니다. 하지만 답변은 판나코타에 대한 역사 설명을 알려줍니다.



이렇듯 Memory 모듈과 함께 사용함으로써 대화 기록을 기반으로 명령을 실행할 수 있습니다. 



---

### ✏️ Wrap up!

이번에는 Memory 모듈과 Agent를 결합하여 대화 기록을 가진 코드를 작성해보았는데요, 단순히 Memory 모듈을 초기화하고 agent에 넣어주니 매우 간단하게 느껴집니다. 

지금까지 유용한 Langchain의 모듈과 그 컨셉들에 대해 열심히 배워보았는데요! 이제 여러 가지 실제 적용할 수 있는 사례에 적용해보는 과정으로 넘어가봅시다!
