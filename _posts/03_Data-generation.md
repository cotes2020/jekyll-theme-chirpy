
프로젝트에서 근무할 때, 테스트 진행을 위한 테스트 데이터 생성도 꽤 큰 일이었습니다.

그 때 mock data를 생성하는 것도 꽤나 큰 일이었는데요 langchain에서는 이러한 가상의 데이터를 생성해주는 작업도 손쉽게 진행해볼 수 있다고 해요.

**Langchain의 `data_generation`을 이용해서 가상의 고객 데이터를 만들어봅시다!**

---

#### 환경

- python 3.12
- pip list
  ```
  pip install --upgrade --quiet  langchain langchain_experimental langchain-openai
  ```


### 1. Setup

랭체인의 라이브러리들을 인스톨해줍니다.

`create_openai_data_generator` 체인을 사용해줄 것이기 때문에, 아래의 요소들이 필요합니다.

```python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_experimental.tabular_synthetic_data.openai import (
    OPENAI_TEMPLATE,
    create_openai_data_generator,
)
from langchain_experimental.tabular_synthetic_data.prompts import (
    SYNTHETIC_FEW_SHOT_PREFIX,
    SYNTHETIC_FEW_SHOT_SUFFIX,
)
from langchain_openai import ChatOpenAI
```


### 2. 데이터 모델 정의

모든 데이터셋에 대해 "스키마"구조를 만들어줍니다. `int`, `str`, `float` 등으로 데이터 타입을 제한해줍니다.

```python
# 1. 데이터 모델 정의
# 데이터셋을 위한 데이터 스키마 정의
class Customer(BaseModel):
    customer_id: int
    customer_name: str
    identification_number: str
    birth: str
    gender: str
    email: str
    phone_number: str
```


### 2. 샘플 데이터

가상 데이터 생성 가이드역할을 해주기 위해, 실제 예시 데이터를 제공해줍니다. 아래 예시들은 데이터 생성 시에 사용될 `seed data`가 됩니다.

이와 비슷한 데이터로 생성될 수 있습니다.

```python
examples = [
    {
        "example": """Customer ID: 1234567890, Customer Name: 김동구, Identification Number: 910101-1211324,
        Gender: 남, Email: dongku12@mockmail.com, Phone Number: 010-1234-1234"""
    },
    {
        "example": """Customer ID: 9876543210, Customer Name: 이수민, Identification Number: 850615-2177542,
        Gender: 여, Email: sumini85@mockmail.com, Phone Number: 010-5678-5678"""
    },
    {
        "example": """Customer ID: 4567891230, Customer Name: 박철수, Identification Number: 001220-3155232,
        Gender: 남, Email: chulsu78@mockmail.com, Phone Number: 010-9876-9876"""
    },
    {
        "example": """Customer ID: 5123215413, Customer Name: 정수정, Identification Number: 061211-4155232,
        Gender: 여, Email: sujeong06@mockmail.com, Phone Number: 010-4124-5125"""
    },
]
```

### 3. 프롬프트 템플릿 작성하기

프롬프트 템플릿을 이용해서 데이터 생성을 위한 가이드를 작성해줍니다. 이 프롬프트를 이용해서 LLM이 어떤 형식에 맞춰 어떤 데이터를 작성해야 할지 정보를 줍니다.

```python
OPENAI_TEMPLATE = PromptTemplate(input_variables=["example"], template="{example}")

prompt_template = FewShotPromptTemplate(
    prefix=SYNTHETIC_FEW_SHOT_PREFIX,
    examples=examples,
    suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
    input_variables=["subject", "extra"],
    example_prompt=OPENAI_TEMPLATE
)

```

`FewShotPromptTemplate`에 대한 설명입니다.

- `prefix`, `suffix`: 언어 모델에게 주어질 프롬프트에 컨텍스트나 지침을 포함

  - `SYNTHETIC_FEW_SHOT_PREFIX`
    - 프롬프트의 시작 부분에 위치
    - 예를 들어, 데이터 생성 작업에 필요한 배경 정보나 모델이 이해해야 할 기본적인 설명이 여기에 포함될 수 있습니다.
    - 이를 통해 모델이 예제 데이터의 형식과 목적을 명확히 이해할 수 있습니다.

  * `SYNTHETIC_FEW_SHOT_SUFFIX`
    * 프롬프트의 끝 부분에 위치
    * 예제 데이터 뒤에 추가적인 지침이나 결론을 제공하는 역할을 합니다.
    * 이는 모델이 예제를 기반으로 더 많은 데이터를 생성할 때 참고할 수 있는 추가적인 정보를 제공합니다.

* `examples`: 정의했던 샘플데이터 할당
* `input_variables`: synthetic_data_generator을 generate할 때, 템플릿으로 사용할 변수
* `example_prompt`: 각 예제의 형식을 정의하는 템플릿, `OPENAI_TEMPLATE`은 프롬프트 내에서 각 개별 예제가 일관된 형식으로 렌더링되도록 하여, 언어 모델에 일관성을 제공


### 4. 데이터 생성기 정의

위에서 정의한 스키마와 프롬프트를 통해, LLM에 전달하여 mock 데이터를 생성할 generator를 정의해줍니다.

```python
synthetic_data_generator = create_openai_data_generator(
    output_schema=Customer,
    llm=ChatOpenAI(model="gpt-4o", temperature=1),
    prompt=prompt_template
)
```


### 5. 데이터 생성

`generate`하여 데이터를 생성해줍니다!

```python
synthetic_results = synthetic_data_generator.generate(
    subject="customer list",
    extra="""
        고객 ID는 반드시 10글자의 숫자여야해. 데이터 간 중복이 있으면 안돼.
        고객명은 반드시 2글자에서 4글자 사이의 한글이어야해.
        identification_number은 주민번호를 의미해. 한국의 주민등록번호 규칙을 따라서 생성해.
        이메일은 이메일 형식을 따라야해.
        전화번호는 데이터 간 중복이 있으면 안돼.
    """,
    runs=15
)

for cust in synthetic_results:
    print(cust)
```

`subject`에는 생성하는 데이터의 주제를, `extra`에는 데이터 생성 시 유의해야 할 부가정보를 넣어줍니다.

`runs`에는 생성할 데이터의 갯수를 줍니다.

```
customer_id=7432189650 customer_name='이현우' identification_number='920304-2257890' birth='1992-03-04' gender='남' email='hyunwoo92@mockmail.com' phone_number='010-8765-4321'
customer_id=7896543210 customer_name='김아름' identification_number='890715-2541386' birth='1989-07-15' gender='여' email='areum89@mockmail.com' phone_number='010-1234-5678'
customer_id=1234567890 customer_name='김민정' identification_number='981230-2157789' birth='1998-12-30' gender='여' email='minjung98@mockmail.com' phone_number='010-3456-7890'
customer_id=6789012345 customer_name='박지성' identification_number='840223-1034567' birth='1984-02-23' gender='남' email='jisung84@mockmail.com' phone_number='010-8765-2314'
customer_id=3748291065 customer_name='박진우' identification_number='920509-1278302' birth='1992-05-09' gender='남' email='jinwoo92@mockmail.com' phone_number='010-9876-5432'
customer_id=4567891230 customer_name='이동훈' identification_number='910825-1234567' birth='1991-08-25' gender='남' email='donghun91@mockmail.com' phone_number='010-2345-6789'
customer_id=9876543210 customer_name='이수민' identification_number='000112-4567890' birth='2000-01-12' gender='여' email='sumin00@mockmail.com' phone_number='010-1234-5678'
customer_id=1234567890 customer_name='김철수' identification_number='850305-1234567' birth='1985-03-05' gender='남' email='cheolsu85@mockmail.com' phone_number='010-9876-1234'
customer_id=2847365920 customer_name='한지민' identification_number='880715-2345673' birth='1988-07-15' gender='여' email='jimin88@mockmail.com' phone_number='010-5678-1234'
customer_id=3748291056 customer_name='박준형' identification_number='931215-1234567' birth='1993-12-15' gender='남' email='junhyung93@mockmail.com' phone_number='010-3456-7890'
customer_id=5678901234 customer_name='정수현' identification_number='921030-2345678' birth='1992-10-30' gender='여' email='soohyun92@mockmail.com' phone_number='010-2345-6789'
customer_id=8912345670 customer_name='이영희' identification_number='790812-2345678' birth='1979-08-12' gender='여' email='younghee79@mockmail.com' phone_number='010-4321-8765'
customer_id=1029384756 customer_name='김하늘' identification_number='900101-1234567' birth='1990-01-01' gender='여' email='haneul90@mockmail.com' phone_number='010-6789-1234'
customer_id=1234567890 customer_name='김민수' identification_number='850315-2345678' birth='1985-03-15' gender='남' email='minsu85@mockmail.com' phone_number='010-9876-5432'
customer_id=5638941250 customer_name='박지민' identification_number='970405-1234567' birth='1997-04-05' gender='남' email='jimin97@mockmail.com' phone_number='010-1234-5678'
```

결과물이 잘 나왔습니다!

심지어 2000년 생 이후의 여자일 경우 주민번호의 시작이 4인것도 잘 인지해서 반환해주었습니다.

---

이제 langchain을 이용해서 테스트에서 사용할 샘플 데이터들을 쉽게 만들어보아요 :) !
