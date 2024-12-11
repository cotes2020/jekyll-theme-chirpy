이번 챕터에서는 여러 가지 모듈들을 서로 연동하기 위한 방법인 `Chains` 모듈에 대해 공부해보겠습니다!

랭체인에서 모듈들을 연동하기 위한 방법은 크게 두 가지로 분류됩니다.

기존 방식이었던 `Legacy Chains 방식`과 현재 권장되고 대체되는 방식인 `LCEL Chains 방식`입니다.

Chains이 무엇인지에 대해 알아보고, 각 방식들에 대해서도 자세히 알아보도록 하겠습니다 :)

---

# Chains

**<mark>`Chains`는 LLM, 도구, 또는 데이터 전처리 단계의 모듈을 하나의 묶음으로 연동하고, 각 단계에 대한 호출 시퀀스를 나타냅니다.</mark>**

![1720574269212](image/02_Langchain모듈-04_Chains_1/1720574269212.png)

Chains는 랭체인의 모듈들을 연결하는 Runnable입니다. 각 모듈들의 input / output type에 맞춰 순서대로 구성해주면 됩니다.

Runnable Interface는 chain을 실행할 수 있는 다양한 메서드를 제공합니다. (`.stream`, `.invoke`, `.batch` 등)

실제로 이렇게 실행하면 내부적으로는 메서드에 따라 runnable을 구현(implements)하고 있는 여러 인터페이스로 실행됩니다.

> 참고
>
> * **`RunnableSequence`** : Chain을 순차적으로 실행합니다. `.stream` 메서드 호출 시 `invoke` 메서드를 사용하여 순차적으로 각 단계의 출력을 스트리밍합니다.
> * **`RunnableParallel`** : 여러 개의 독립적인 Chain들을 동시에 병렬로 실행합니다. 병렬 실행을 위해 여러 입력을 동시에 처리하며, 각 입력의 출력을 스트리밍합니다.
> * **`RunnableMap`** : 입력 리스트를 받아 각각을 처리하여 출력 리스트를 생성합니다. `.stream`을 사용하여 각 입력에 대한 출력을 순차적으로 스트리밍할 수 있습니다.
> * **`RunnableLambda`** : 람다 함수를 사용하여 입력을 처리합니다. `.stream`을 사용하면 람다 함수의 출력을 실시간으로 스트리밍할 수 있습니다.

- Langchain Components' input/output Type

![1720574388998](image/02_Langchain모듈-04_Chains_1/1720574388998.png)

LangChain의 체인은 LCEL 방식과 legacy 방식으로 분류됩니다.

**LCEL 방식**

- [LCEL](https://js.langchain.com/v0.1/docs/expression_language/)은 LangChain Expression Language의 줄임말로 랭체인에서 제공하는 고유의 표현 방식입니다.
- LCEL 체인은 최신 방식으로, 언어 모델(LLM)을 기반으로 체인을 설계하고 실행합니다.
- 유연성과 확장성을 강조하며, 코드의 가독성과 유지보수를 용이하게 합니다.

**Legacy 방식**

- 기존 체인 방식으로, 다양한 상황에 맞는 여러 체인 유형이 있습니다.
- 각 체인은 특정 작업에 최적화되어 있으며, 필요에 따라 API 호출, 데이터베이스 쿼리, 문서 처리 등을 수행합니다.

---

## Legacy Chains

먼저 기존 방식에 대해 간단하게 알아봅시다.

링크를 따라 이동하면 현재의 모든 [레거시 체인의 목록](https://js.langchain.com/v0.1/docs/modules/chains/#legacy-chains)을 볼 수 있습니다. 많은 기능들이 있지만, 이 중 세 가지의 체인 모듈을 살펴보겠습니다.

1. **`LLMChain`, `ConversationChain` : 여러 모듈의 조합**

   <img src="uploads/02_Langchain모듈-04_Chains/2024-06-05-11-02-48-image.png" title="" alt="" data-align="left">

   Chain이 없어도 구현을 할 수 있지만 Chain을 이용하면 여러 단계를 하나의 모듈로 대체할 수 있습니다.

   우리는 사실 그 전에도 Chain 모듈을 사용하여 구현을 했습니다. [Memory를 이용한 챗봇 구현](https://git.bwg.co.kr/gitlab/study/langchain/-/wikis/02_Langchain%EB%AA%A8%EB%93%88-03_Memory_1_chatbot#conversationchain-%EB%AA%A8%EB%93%88-%EC%9D%91%EC%9A%A9%ED%8E%B8) 시, Memory 모듈에서 각각 구현했던 모듈들을 `ConversationChain` 하나의 모듈을 이용해 변환하여 구현해보기도 했습니다.
2. **`LLMRequqestChain(deprecated)`,`createOpenAIChain` : 특정 용도에 특화된 체인**

   <img src="uploads/02_Langchain모듈-04_Chains/2024-06-05-11-03-28-image.png" title="" alt="" data-align="left">

   언어 모델의 호출만으로는 대응하기 어려운 기능이나 복잡한 처리를 랭체인 측에서 미리 내장해 특정 용도에 특화된 Chains도 존재합니다.

   현재는 deprecated되었지만 `LLMRequqestChain`의 경우 주어진 URL에 접속해 얻은 결과와 질문을 조합해 만든 프롬프트로 언어 모델을 호출할 수 있습니다. `createOpenAIChain`은 OpenAPI 스펙을 검색하여 체인을 생성할 수 있습니다.
3. **`SimpleSequentialChain` : 체인들을 묶음**

   <img src="uploads/02_Langchain모듈-04_Chains/2024-06-05-11-03-40-image.png" title="" alt="" data-align="left">

   Chain은 하나의 `기능 덩어리` 라고 할 수 있습니다. 이 기능 덩어리를 여러 개 순비해 순서대로 실행하거나 필요에 따라 호출할 수 있도록 Chains 자체를 묶을 수 있습니다. 예를 들어 `LLMRequqestChain`으로 웹페이지에서 얻은 정보를 요약하고, 그 정보를 다른 Chain으로 처리하는 등의 작업이 가능하다.

## LCEL Chains

`LCEL Chains`방식의 장점은 아래와 같습니다.

- 체인의 내부를 수정하고 싶을 때 LCEL을 간단히 수정할 수 있기 때문에 유용합니다.
- 기본적으로 스트리밍, 비동기 및 배치를 지원합니다.
- 각 단계의 가시성(observability)이 높습니다. 즉, 중간 단계의 결과에 대한 결과 접근이 용이합니다.

예시를 통해 위 장점들을 확인해볼까요?

### 기본 예시: 프롬프트 + 모델 + 출력 파서

[LCEL의 다양한 common task 예제](https://python.langchain.com/v0.2/docs/how_to/#langchain-expression-language-lcel)들이 있습니다. 그 중 가장 기본적이고 일반적인 사용 사례는 prompt 템플릿과 모델을 함께 연결하는 것 입니다. 이것이 어떻게 작동하는지 보기 위해, 간단한 Chain 예제를 구현해 보겠습니다.

> lcel.py

```python
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Template 정의
template = "{subject}에 관련된 재밌는 농담을 말해줘."
# from_template 메서드를 이용해 PromptTemplate 객체 생성
prompt = PromptTemplate.from_template(template)
# -> 아래와 동일한 코드이지만 위처럼 간결하게 표현할 수 있습니다.
# prompt = PromptTemplate(
#     template = "{subject}에 관련된 재밌는 농담을 말해줘.",
#     input_variables=[
#         "subject"
#     ]
# )

# 모델 연결
model = ChatOpenAI(
    model="gpt-4o"
)

# Output Parser 정의
output_parser = StrOutputParser()


# 프롬프트, 모델, 출력 파서를 연결하여 처리 체인을 구성합니다.
# chain = prompt | model
chain = prompt | model | output_parser
# 완성된 Chain 을 이용하여 subject 를 '코끼리'으로 설정하여 실행합니다.
print(chain.invoke({"subject": "코끼리"}))

```

먼저 Template과 Model, Output Parser 모듈을 각 각 정의해줍니다.

이제 세 개의 모듈을 엮어줄 Chain을 생성해봅시다. 여기서 우리는 LCEL을 사용하여 다양한 구성 요소를 단일 체인으로 결합합니다.

```ini
chain = prompt | model | output_parser
```

`|` 기호는 [unix 파이프 연산자](https://en.wikipedia.org/wiki/Pipeline_(Unix))와 유사하며, 서로 다른 구성 요소를 연결하고 **한 구성 요소의 출력을 다음 구성 요소의 입력으로 전달**합니다.

이 체인에서 사용자 입력은 프롬프트 템플릿으로 전달되고, 그런 다음 프롬프트 템플릿 출력은 모델로 전달되며, 그 다음 모델 출력은 출력 파서로 전달됩니다. 각 구성 요소를 개별적으로 살펴보면 무슨 일이 일어나고 있는지 이해할 수 있습니다.

1. 원하는 주제에 대한 사용자 입력을 `{"subject": "코끼리"}`로 전달합니다.
2. `prompt` 컴포넌트는 사용자 입력을 받아 `subject`을 사용하여 프롬프트를 구성한 후 PromptValue를 생성합니다.

   - prompt 출력값 = model 입력값

     ```ini
     text='코끼리에 관련된 재밌는 농담을 3가지 말해줘.'
     ```
3. `model` 컴포넌트는 생성된 프롬프트를 가져와 OpenAI LLM 모델에 평가를 위해 전달합니다. 모델에서 생성된 출력은 `ChatMessage` 객체입니다.

   - model 출력값 = output_parser 입력값

     ```ini
     content='물론이죠! 여기 코끼리에 관련된 농담 하나 있어요:\n\n왜 코끼리는 컴퓨터를 쓰지 않을까요?\n\n왜냐하면 그들은 마우스를 너무 무서워하거든요! 🐘🖱️😄' response_metadata={'token_usage': {'completion_tokens': 55
     , 'prompt_tokens': 23, 'total_tokens': 78}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_4008e3b719', 'finish_reason': 'stop', 'logprobs': None} id='run-a9d82618-9680-45b4-a123-7dfe2ca36a83-0'
     ```
4. 마지막으로, `output_parser` 컴포넌트는 `ChatMessage`를 받아 이를 Python 배열로 변환하며, 이는 invoke 메서드에서 반환됩니다.

   - output_parser 출력값 = 최종 chain의 출력값

     ```ini
     물론이죠! 여기 코끼리에 관한 재밌는 농담이 있습니다:

     왜 코끼리는 노트북을 사용하지 않을까요?

     왜냐하면, 마우스를 너무 무서워하거든요! 🐘🖱️😄
     ```

LCEL 방식은 마치 블록을 조립하는 것 같아서, 중간 과정의 디버깅도 쉽고 각 모듈을 조립하기도 쉬울 것 같습니다!

### LCEL 인터페이스

사용자 정의 체인을 가능한 쉽게 만들 수 있도록, [&#34;Runnable&#34;](https://api.python.langchain.com/en/stable/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable) 프로토콜을 구현되어있습니다. `Runnable` 프로토콜은 대부분의 컴포넌트에 구현되어 있습니다.

이는 표준 인터페이스로, 사용자 정의 체인을 정의하고 표준 방식으로 호출하는 것을 쉽게 만듭니다.

#### 표준 인터페이스 종류

위 예제에서는 `invoke`라는 메서드를 이용하여 체인을 호출하고 답변을 받는 방식을 사용해보았습니다. LCEL에서는 `invoke` 외에도 여러 인터페이스를 제공합니다.

표준 인터페이스는 다음의 종류가 있습니다.

- [`stream`](https://js.langchain.com/v0.1/docs/expression_language/interface/#stream): 응답의 일부를 스트리밍으로 반환합니다.
- [`invoke`](https://js.langchain.com/v0.1/docs/expression_language/interface/#invoke): 입력에 대해 체인을 호출합니다.
- [`batch`](https://js.langchain.com/v0.1/docs/expression_language/interface/#batch): 입력 목록에 대해 체인을 호출합니다.
- [`streamLog`](https://js.langchain.com/v0.1/docs/expression_language/interface/#stream-log): 최종 응답 외에 중간 단계가 발생할 때마다 스트리밍으로 반환합니다.
- [`streamEvents`](https://js.langchain.com/v0.1/docs/expression_language/interface/#stream-events): 체인에서 발생하는 이벤트를 스트리밍으로 반환하는 베타 기능 (introduced in `@langchain/core` 0.1.27)

위에서는 다루지 않았지만, 자주 사용되는 인터페이스 `stream`과 `batch`에 대해 간단히 살펴보도록 하겠습니다.

- **stream**

  데이터 스트림을 생성하고, 이 스트림을 반복하여 각 데이터의 내용(`content`)을 즉시 출력합니다. `end=""` 인자는 출력 후 줄바꿈을 하지 않도록 설정하며, `flush=True` 인자는 출력 버퍼를 즉시 비우도록 합니다. 이는 스트리밍 데이터를 실시간으로 처리할 때 유용하게 사용됩니다.

  ```python
  # 프롬프트, 모델 연결하여 처리 체인을 구성합니다. (output_parser를 제외하였습니다.)
  chain = prompt | model

  # chain.stream 메서드를 사용하여 '멀티모달' 토픽에 대한 스트림을 생성하고 반복합니다.
  for s in chain.stream({"subject": "코끼리"}):
      # 스트림에서 받은 데이터의 내용을 출력합니다. 줄바꿈 없이 이어서 출력하고, 버퍼를 즉시 비웁니다.
      print(s, end="", flush=True)
  ```

  코드를 수행하면 답변이 완성된 후 완성된 답변을 출력받는 것이 아니라, 실시간으로 생성되는 답변을 출력받을 수 있습니다.
- **batch**

  여러 개의 딕셔너리를 포함하는 리스트를 인자로 받아, 각 딕셔너리에 있는 `subject` 키의 값을 사용하여 일괄 처리를 수행합니다. 이 예시에서는 두 개의 주제, `개구리`와 `병아리`에 대한 처리를 요청합니다.

  ```python
  # 프롬프트, 모델, 출력 파서를 연결하여 처리 체인을 구성합니다.
  chain = (prompt | model)
  # 주어진 주제 리스트를 batch 처리하는 함수 호출
  print(chain.batch([{"subject": "개구리"}, {"subject": "코끼리"}]))
  ```

  코드를 수행하면 개구리에 대한 답변과 코끼리에 대한 농담을 각 각 `AIMessage` 객체로 반환받는 것을 확인할 수 있습니다.
- **`RunnableParallel` 구현해보기**

  지금까지는 `RunnableSequence`를 구현하는 방법에 대해서만 알아보았는데요,
  Runnable의 interface method의 종류는 아니지만, Runnable을 상속받아 구현되는 `RunnableParallel`을 간단히 구현해보겠습니다.

  하나의 체인을 더 만들어봅시다.

  ```python
  # 새로운 프롬프트 생성
  prompt2 = PromptTemplate.from_template("{subject}에 관련된 시를 알려줘.")

  # 새로운 체인 생성
  chain2 = prompt2 | model | output_parser
  ```

  `RunnableParallel`을 이용해서 체인 두 개를 서로 연결해줍니다. 그 뒤에 동시에 수행시켜줍니다.

  ```python
  from langchain_core.runnables import RunnableParallel

  chain = prompt | model | output_parser
  chain2 = prompt2 | model | output_parser

  map_chain = RunnableParallel(
     joke=chain, 
     poem=chain2, 
  )

  print(map_chain.invoke({"subject": "개구리"}))
  ```

  ```output
  {'joke': 
      '물론! 여기 개구리에 관한 재밌는 농담이 있어요:
      왜 개구리는 시험을 잘 보지 못할까요?
      왜냐하면, 그는 모든 문제를 "개굴개굴" 대답하기 때문이에요!
      혹시 더 필요하신 농담이 있으면 말씀해 주세요!', 
   'poem': 
      '개구리에 관한 시로는 유명한 한국의 시인 조지훈의 "승무"에서 개구리가 등장하는 부분을 소개할 수 있습니다. 그러나 "승무"는 개구리만을 주제로 한 시는 아닙니다. 또한 개구리를 주제로 한 시 중에서 잘 알려진 외국 시인들의 작품도 몇 가지 소개할 수 있습니다. 
      예를 들어, 일본의 하이쿠 시인 마쓰오 바쇼(Matsuo Basho)의 유명한 하이쿠가 있습니다.

      예를 들어, 마쓰오 바쇼의 하이쿠를 소개해드리겠습니다:\n\n```\n오래된 연못\n개구리 한 마리\n뛰어든다, 물 소리

      ```원문:```

      \n古池や\n蛙飛びこむ\n水の音\n```\n\n이 하이쿠는 자연의 순간을 매우 간결하게 포착한 작품으로, 개구리가 뛰어드는 순간의 소리를 통해 고요함 속의 움직임을 표현한 시입니다. 이 시는 일본 문학에서 매우 중요한 위치를 차지하고 있으며, 많은 사람들에게 사랑받고 있습니다.'
   }
  ```

  [LangSmith](https://smith.langchain.com/public/4a4ab030-8a92-46d0-b6ee-60c37fe2190e/r)의 결과에서도 `RunnableParallel`이 구현되어 실행된 것을 볼 수 있습니다.

---

### ✏️ Wrap up!

이번 시간에는 AI 어플리케이션을 개발하기 위해 사용되는 모듈들을 간단하게 연결해주는 `LCEL` chain에 대해 알아보았습니다.

이제 LangChain에서 제공하는 대부분의 모듈에 대해 이미 학습하셨습니다! :)

다음은 단순히 언어모델과의 텍스트를 통한 송수신을 넘어선 더 다양한 작업의 수행을 가능하게 해주는 `Agents` 모듈에 대해 공부해보도록 하겠습니다!
