---
title: [Langchain Study | 1. Basic]
categories: [AI, Langchain]
tags: [Langchain, AI, AI Application]		
---

Langchain이란 무엇일까요?

앞으로 글에서는 여러 챕터로 나눠 Langchain이 무엇이고, Langchain을 사용하는 방법, 더 나아가 응용하는 방법에 대해 배워보도록 하겠습니다.

---

# LLM

## LLM이란?

Langchain이 무엇인지에 대해 설명하기 앞서 우리는 LLM에 대한 이해가 필요합니다.

<mark>LLM은 Large Language Models의 줄임말로 **대형 언어 모델**이라는 의미</mark>입니다.

최근 AI와 관련된 많은 용어들이 등장하면서, 여러 용어들이 혼재되어 사용되는 경우가 많습니다. 이러한 관계를 아래와 같은 그림으로 정리할 수 있습니다

![AI와LLM]({{"/assets/img/posts/image.png"  | relative_url }})

`출처: <랭체인으로 LLM 기반의 AI 서비스 개발하기>  (서지영)`

- AI (인공지능): 기계가 인간의 지능을 모방하는 모든 기술을 포함하는 가장 넓은 범주입니다.
- 머신러닝 (Machine Learning): 데이터를 통해 학습하는 시스템에 초점을 맞춘 AI의 하위 분야입니다.
- 인공 신경망 (Artificial Neural Networks): 인간의 뇌 구조와 기능에서 영감을 받은 머신러닝의 하위 분야입니다.
- 딥러닝 (Deep Learning): 여러 층의 신경망을 사용하는 신경망의 하위 분야로, 매우 복잡한 모델을 생성할 수 있습니다.
- 생성형 AI (Generative AI): 딥러닝 기술을 사용하여 텍스트, 이미지 등 콘텐츠를 생성할 수 있는 AI입니다.
- LLM (대형 언어 모델): 인간 언어를 처리하고 생성하는 데 중점을 둔 특정 생성형 AI입니다.

우리에게 ChatGPT는 매우 익숙합니다. 이러한 OpenAI의 ChatGPT, 구글의 Gemini, 엔트로픽의 Claude는 GenAI 즉, 생성형 AI입니다. LLM을 활용하여 이러한 생성형 AI와 같은 어플리케이션을 만들기도 합니다. 물론 LLM은 생성형 AI에 활용되는 것 외에 다른 여러 분야에서도 활용될 수 있습니다. 번역 서비스, 교육, 법률 분야 등에서 넓게 사용되고 있습니다.

참고로 LLM의 종류는 아래 사진처럼 매우 많습니다.
![llm-모델-종류]({{"/assets/img/posts/image-3.png"  | relative_url }})

## LLM 활용 사례

| Tasks           | 설명                                             |
| --------------- | ---------------------------------------------- |
| **음성-텍스트 변환**   | 음성 데이터를 텍스트로 변환                                |
| **기계번역**        | 한 언어의 텍스트를 다른 언어로 변환                           |
| **페리프레이징**      | 주어진 텍스트의 의미를 유지하면서 다른 방식으로 표현                  |
| **질문응답**        | 주어진 질문에 대한 응답 생성                               |
| **텍스트 분류**      | 텍스트를 사전 정의된 카테고리 중 하나로 분류 - 감정분석, 주제 분류, 스팸 감지 |
| **개체명 인식(NER)** | 텍스트에서 특정 유형의 정보를 식별 `예) 인물, 지역, 시간`            |
| **텍스트 요약**      | 긴 텍스트를 짧게 요약                                   |
| **텍스트 생성**      | 주어진 입력에 기반한 새로운 텍스트를 생성                        |
| **소스코드 설명**     | 소스코드를 자연어로 설명하는 문장 생성                          |



**💡필수 용어!**

> - **Prompt**
>   
>   - 사용자가 언어 모델에 제공하는 입력
>   
>   - 질문, 문장, 단어, 문서의 일부 등 다양한 형태
> 
> - **Completion**
>   
>   - 언어 모델이 주어진 Prompt에 의해 생성한 응답
>   
>   - 모델이 학습한 내용을 바탕으로 생성하는 텍스트(이미지, 문서 등..)



---

# LangChain

## Langchain이란?

**<mark>Langchain은 LLM(언어 모델) 기반의 AI 어플리케이션을 만들 때, 보다 쉽게 만들 수 있도록 도와주는 프레임워크입니다.</mark>**

Langchain을 사용하지 않고도 LLM을 사용한 어플리케이션을 개발할 수 있지만, 시간이 많이 걸리고 복잡합니다. 

아래의 영상에서의 어플리케이션을 만들어봅시다.
chatGPT LLM을 이용하여, 질문에 대한 답을 생성하는 웹 어플리케이션입니다.

![llm-app]({{"/assets/img/posts/langchain-app.gif"  | relative_url }})

langchain을 이용한다면 단 18줄의 코드만으로 front-end부터 back-end까지 개발이 완료됩니다.

```python
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

st.title('🦜🔗 Quickstart App')

openai_api_key = st.sidebar.text_input('OpenAI API Key')
def generate_response(input_text):
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.7, openai_api_key=openai_api_key)
    st.info(llm([HumanMessage(content=input_text)]).content)

with st.form('my_form'):
    text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-'):
        generate_response(text)
```

아래의 그림은 어플리케이션의 구조를 단순화한 그림입니다.
![alt text]({{"/assets/img/posts/image-6.png"  | relative_url }})
여기서 랭체인의 역할을 중점으로 본다면, 화면에서 넘어오는 질문을 입력받아 모델에 전달하고, 모델로부터 응답을 받아 다시 화면에 보내주는 역할을 합니다. 사실 랭체인은 다양한 모듈들을 가지고 있으며 이를 통해 더 많은 확장성을 가집니다. 

### Langchain의 모듈

Langchain은 LLM 기반의 어플리케이션 개발 시, 아래 나열된 다양한 모듈과 도구를 통해 개발 과정을 단순화하고 효율화합니다. 

![]({{"/assets/img/posts/2024-06-24-16-05-26-image.png" | relative_url }})

Langchain의 모듈을 이용하면 각 과정들을 별도로 구현할 필요 없이 구현할 수 있으며, 각 과정을 서로 연동(체인화)하는데 매우 편리합니다.

각 모듈에 대해 간단히 살펴보겠습니다.

1. **언어 모델 통합 및 관리**

LangChain은 OpenAI, HuggingFace Hub와 같은 다양한 LLM 모델을 통합하고 관리할 수 있는 인터페이스를 제공합니다. 이를 통해 개발자는 여러 모델을 쉽게 전환하고 사용할 수 있습니다.

2. **프롬프트 관리**

LangChain은 프롬프트 템플릿을 제공하여 사용자 입력을 효과적으로 관리하고, 모델의 출력을 구조화하는 출력 파서를 통해 결과를 적절하게 포맷할 수 있게 합니다. 이를 통해 일관된 입력과 출력을 유지할 수 있습니다.

3. **체인**

체인은 말 그대로 연결 고리를 만드는 것입니다. 이는 랭체인의 워크플로우의 핵심입니다. 체인을 통해 LLM을 다른 구성 요소와 결합하여 일련의 태스크를 실행함으로써 어플리케이션을 생성합니다. 이를 통해 복잡한 작업을 단계별로 구성하고 관리할 수 있습니다.

4. **인덱스**

LLM 자체에 학습되지 않은 특정 외부 데이터 소스(private 문서, 이메일 등)를 활용할 수 있도록 해주는 것입니다. 인덱스 기능을 이루는 요소로 `도큐먼트 로더`, `벡터 데이터베이스`, `텍스트 스필리터` 등이 있습니다.

5. **메모리 기능**

LangChain은 단기 및 장기 메모리 기능을 통해 체인이나 에이전트가 사용자와의 이전 상호작용을 기억할 수 있게 합니다. 이는 지속적인 대화형 애플리케이션에서 중요한 역할을 합니다.

6. **에이전트 역할**

LangChain의 에이전트는 입력에 따라 사용할 도구나 데이터를 결정하고, 적절한 행동을 선택합니다. 이는 복잡한 의사결정 과정을 자동화하는 데 유용합니다.

7. **콜백 기능**

LangChain은 특정 지점에서 수행할 함수를 트리거하는 콜백 기능을 제공합니다. 이를 통해 LLM 실행 중에 다양한 작업을 자동화할 수 있습니다.

## LangChain 활용 사례

랭체인을 사용해 챗봇 또는 개인 비서를 만들고, 문서 또는 구조화된 데이터에 대한 Q&A를 요약, 분석, 생성하고, 코드를 쓰거나 이해하고, API와 상호작용하고, 생성형 AI를 활용하는 여러 애플리케이션을 만들 수 있습니다. 

#### Case 1 : ChatBot

- 사용자가 질문을 하면 LLM이 답변을 하고, LLM으로부터 반환받은 답변을 다시 사용자에게 전달하는 어플리케이션

#### Case 2 : Q&A with RAG

- LLM에는 존재하지 않는 데이터를 추가적으로 더하여, 특정 정보에 대한 Q&A 어플리케이션

- [DucuMentor AI](https://docuai.bwg.co.kr/)
  
  - BwG 사내 문서를 임베딩하여, 검색결과를 AI로 정리하여 전달하는 Q&A 어플리케이션

#### Case 3: Tagging

<img src="https://python.langchain.com/v0.1/assets/images/tagging-93990e95451d92b715c2b47066384224.png" title="" alt="Image description" data-align="center">

- 문서들을 분류 체계에 따라 라벨링하는 것
  
  >  **분류체계**
  > 
  > - 감정
  > 
  > - 언어
  > 
  > - 스타일(친근, 포멀, 매우 자세하게, 요약하여.. )
  > 
  > - 정치 경향성
  > 
  > - 주제 
  
  <details>
  <summary>예제</summary>
  
  ![]({{"/assets/img/posts/2024-06-24-15-45-44-image.png"  | relative_url }})  
  
  </details>

#### Case 4 : Web scraping

- Web Site의 URL을 가지고, 정보들을 Scraping 해오는 기능 구현

- ex)
  
  - 월스트리트저널 홈페이지의 URL을 입력 → 새로운 기사의 이름과 요약 추출
  
  - 검색된 자료를 자동으로 retriever로 이용 이를 다시 RAG로 할당
  
  - 월스트리트저널의 기사들을 바탕으로 한 Q&A 어플리케이션 개발



[LangChain의 Use Cases](https://python.langchain.com/v0.1/docs/use_cases/)



---

**❗알고 있으면 좋은 정보**

- LangChain 공식 문서
   - [Concept](https://python.langchain.com/v0.1/docs/get_started/introduction)
   - [Component](https://python.langchain.com/v0.1/docs/modules/)

- [Chat모델과 Completion 모델의 차이](https://forum.bwg.co.kr/t/chat-completion/197)

---

### ✏️ Wrap up!

아직은 각 기능에 대한 설명이나 과정이 와닿지 않겠지만 차차 소스를 보며 과정 하나하나 본 후에 다시 보시면 위 설명이 쉽게 다가오실거니 이번 챕터는 부담 없이 읽으셔도 됩니다!

이번 챕터에서는 LLM이란 무엇인지에 대한 개념과 LLM을 활용한 어플리케이션을 개발하기 위한 프레임워크로써의 랭체인에 대해 간단하게 알아보았습니다. 또한 랭체인의 다양한 모듈들이 어떤 것이 있는지에 대해 간단히 살펴보았는데요, 다음 챕터부터는 각 모듈들에 대한 좀 더 상세한 설명과 예시 코드들을 소개해드리겠습니다.