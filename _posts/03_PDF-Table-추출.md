포럼에 작성하였던 [Pdf파일에서 테이블 내용 추출하기](https://forum.bwg.co.kr/t/pdf/191)를 활용해서, BXCM 개발 가이드를 기반으로 RAG application을 개발해보겠습니다.

---

### 테스트 개요

- **목적** : [BXCM_Developer_Guide_Development_Standard PDF](./uploads/BXCM_Developer_Guide_Development_Standard.pdf)의 14 페이지의 `[표 2.2] 온라인 어플리케이션 용어` 테이블에 대한 질문 처리를 잘 하도록 한다.
  
  ![](uploads/03_PDF%20내%20Table%20추출하여%20RAG%20APP%20만들기/2024-06-18-10-30-12-image.png)
  
  위 테이블을 테스트 대상으로 선정한 이유는 긴 테이블에 대해서도 잘림 없이 처리하는지 테스트하기 위함이다.

- **과정**
  
  1. PDF 테이블 별도 미처리 테스트
     
     - PDF 파일 로딩 시, 테이블을 별도로 처리하지 않고 단순히 `PyPDFLoader`와 `RecursiveCharacterTextSplitter`을 이용해서 로드, 임베딩, 벡터 DB 저장을 수행한다. => `faiss_index`로 저장
     
     - `faiss_index`를 통해 RAG 기법의 LLM app을 구현하고, `온라인 어플리케이션 용어에 대해 설명해줘`라고 질문을 수행한다.
  
  2. PDF 테이블 처리 후 테스트 (시도 1, 2)
     
     - PDF 파일 로딩 시, 테이블을 별도로 처리 후, 임베딩, 벡터 DB 저장을 수행한다. => `faiss_table_index`로 저장
     
     - `faiss_table_index`를 통해 RAG 기법의 LLM app을 구현하고, `온라인 어플리케이션 용어에 대해 설명해줘`라고 질문을 수행한다.
     
     - PDF 테이블 미처리와 처리 비교

---

### PDF 테이블 별도 미처리 테스트

#### 사전작업: PDF 파일 기반의 벡터 데이터 베이스 구축

우선 비교군을 위해 PDF 파일을 로딩하여 단순히 임베딩, 벡터 DB 저장하는 방법으로 테스트를 수행한다.

> prepare_org.py

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


# 사전준비: PDF 파일 기반의 벡터 데이터 베이스 구축
# => PDF의 테이블 정보를 정제하지 않고, 텍스트로 추출하는 방법


# 1. PDF에서 문장 불러오기
loader = PyPDFLoader("../BXCM_Developer_Guide_Development_Standard.pdf")
pages = loader.load()
# print(len(pages))

# 2. 문장 나누기
# splitter 초기화
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, separators=["\n"]
)

print(pages)

# 로드한 문장 나누기
splitted_documents = text_splitter.split_documents(pages)
# print(len(texts))


# 3. 분할된 문장을 임베딩하여 데이터 베이스 저장
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(splitted_documents, embeddings)
db.save_local("faiss_index") # Replace faiss_index with name of your choice

print("데이터베이스 생성이 완료되었습니다.")
```

분할 방식으로 사용한 `Recursive Character Text Splitter` 는 다음과 같은 특성이 존재한다.

> - **계층적 분할**: 이 방식은 여러 구분자를 사용해 계층적으로 텍스트를 분할합니다. 가장 중요한 구분자(예: 이중 줄 바꿈)부터 시작해 점진적으로 더 세밀한 구분자(예: 단일 줄 바꿈, 공백 등)로 나누어갑니다. 이렇게 하면 의미적으로 관련 있는 텍스트를 최대한 한 덩어리로 유지할 수 있습니다.
> 
> - **의미 보존**: 관련 있는 내용을 함께 유지하려고 노력하기 때문에, 각 청크가 의미 있고 문맥적으로 유효하게 유지됩니다. 이는 효과적인 벡터 검색 및 검색 응용 프로그램을 위해 중요합니다.
> 
> - **적응성**: 재귀적인 특성 덕분에 다양한 문서 형식과 구조에 적응할 수 있습니다. 복잡한 구조의 문서도 효과적으로 처리할 수 있어 각 청크가 유의미하고 유용하게 유지됩니다.

이러한 특성으로 인해 `Recursive Character Text Splitter`는 기술 매뉴얼, 학술 논문, 책과 같이 복잡한 구조와 다양한 형식을 가진 문서에 이상적이기 때문에 `Recursive Character Text Splitter`을 적용하였다.

벡터 스토어로 `FAISS`를 선택한 이유는 python 3.12 버전에서는 chromaDB를 아직 지원하지 않기 때문에 FAISS 를 사용하였다.

#### 답변 생성 테스트

임베딩된 `faiss_index`을 통해 답변 생성 테스트를 진행한다.

<details>
<summary>
<strong>[검색 및 프롬프트 구축 코드] query.py</strong> 
</summary>

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage


# 4. 벡터 데이터베이스에서 검색 실행하기
# 벡터 DB 로드
embeddings = OpenAIEmbeddings()
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
# db = FAISS.load_local("faiss_table_index", embeddings, allow_dangerous_deserialization=True)

# 유사성 검색
# query = "테스트 케이스를 생성할 때 지켜야할 규칙을 알려줘."
# query = "Controller, Service, Component 클래스의 멤버 변수로 사용되는 변수 명명 규칙에 대해 설명해줘."
# query = "어노테이션에 대해 설명해줘"
query = "온라인 어플리케이션 용어에 대해 설명해줘"
documents = db.similarity_search(query)
for d in documents:
    print("# doc => ", d)


# 5. 검색 결과와 질문을 조합하여 LLM에 프롬프트 전달
# 프롬프트 템플릿 정의
documents_string = ""

for document in documents:
    documents_string += f"""
---------------------------
{document.page_content}
"""  # ← 문서 내용을 추가

prompt = PromptTemplate(  # ← PromptTemplate를 초기화
    template="""문장을 바탕으로 질문에 답하세요.

문장:
{document}

질문: {query}
""",
    input_variables=["document", "query"],  # ← 입력 변수를 지정
)

# 모델 초기화
chat = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,
)

result = chat.invoke([
    HumanMessage(content=prompt.format(document=documents_string, query=query))
])

print(result.content)
```

</details>

> **결과**

- 검색된 관련 문서

```
# doc =>  page_content='2. 명명규칙   142.2.1 온라인 어플리케이션\n[표 2.2]\xa0온라인 어플리케이션 용어\n구분 설명\n어플리케이션 •사용자의 요청을 처리하기 위한 프로그램의 집합으로 데이터 액세스와 처리를 위해\n작성된 SQL, 비즈니스 로직 구현을 위한 클래스, 설정파일들을 포함하여 구성된다.(중략).\nIO •메시지(JSON, 고정길이 데이터, XML 등) 형태로 변환할 수 있는 기능을 제공하는\n프레임워크 표준 데이터 전달 객체로 Controller 요청/응답, 대내외계 인터페이스,' metadata={'source': '../BXCM_Developer_Guide_Development_Standard.pdf', 'page': 20}
# doc =>  page_content='2. 명명규칙   182.4 온라인 어플리케이션 명명규칙...(생략)' metadata={'source': '../BXCM_Developer_Guide_Development_Standard.pdf', 'page': 6}
```

- LLM 답변

```markdown
온라인 어플리케이션 용어에 대한 설명은 다음과 같습니다:

1. **어플리케이션**:
   - 사용자의 요청을 처리하기 위한 프로그램의 집합으로, 데이터 액세스와 처리를 위해 작성된 SQL, 비즈니스 로직 구현을 위한 클래스, 설정파일들을 포함하여 구성됩니다.
   - Eclipse의 BX 클라우드 프로젝트로 작성됩니다.
   - 본 프로젝트의 업무 개발자는 별도로 어플리케이션을 생성하지 않고 이미 작성되어 있는 Git Repositor에 제공된 어플리케이션을 Clone하여 사용합니다.
   - 외부에서 식별하기 위해 시스템 내에서 유일한 이름을 가집니다.

2. **컨트롤러 (Controller)**:
   - Rest Client(사용자 화면, 대내외채널 등 외부에서 호출된 사용자 요청)의 요청에 대응되는 클래스입니다.
   - HTTP 클라이언트 요청을 처리(HTTP 의존적인 영역을 처리)합니다.
   - 트랜잭션 처리 단위의 기준이 됩니다.
   - 비즈니스 로직을 처리하기 위해 Service를 호출합니다.
   - Component와 DBIO는 직접 처리하지 않습니다.

3. **서비스 (Service)**:
   - Controller와 1:1로 매핑되는 클래스입니다.
   - Controller에서 호출되는 단위의 기능을 Service의 메소드로 구현합니다.

4. **컴포넌트 (Component)**:
   - 업무 기능 중 공통적으로 사용할 수 있는 기능을 추상화하여 제공하는 클래스입니다.
   - 재사용이 가능한 업무 단위라고 볼 수 있습니다.

5. **DBIO**:
   - 데이터 액세스를 위해 작성하는 개발 리소스로 자바 Interface와 SQL Mapper(.dbio) 파일로 구성됩니다.
   - 개발도구에서 DBIO를 생성하여 개발합니다.
   - SQL은 DBIO에서 SQL ID를 추가하여 작성합니다.

6. **SQL ID**:
   - DBIO에서 생성되는 쿼리를 나타내는 식별자입니다. DBIO 인터페이스의 메소드 이름이 됩니다.

7. **IO**:
   - 메시지(JSON, 고정길이 데이터, XML 등) 형태로 변환할 수 있는 기능을 제공하는 프레임워크 표준 데이터 전달 객체로, Controller 요청/응답, 대내외계 인터페이스에 사용됩니다.
```

#### 📌 결과 분석

결과를 보면 표를 제대로 인식하고 있는 것을 확인할 수 있다.

다만 원문에서는 온라인 어플리케이션 용어 테이블이 Junit 테스트 케이스까지 있는 반면, embedding된 chunk에서는 chunk의 크기 제한에 의해 IO의 중간설명까지 잘려서 7개의 항목까지만 설명해주는 것을 확인할 수 있다.

PDF 테이블 구조를 잘 인지하도록 하여 위의 한계를 해소해보자.

---

## PDF 테이블 처리 후 테스트

PDF 파일을 로딩하여, 텍스트와 테이블을 구분하여 추출한 뒤 원본 PDF의 내용에 맞게 추출된 텍스트와 테이블의 내용을 정렬한다.

그 뒤에 문장을 splitter하고, 분할된 문장을 임베딩하여 벡터 데이터 베이스에 저장한다.

해당 벡터 데이터 베이스를 기반으로 RAG에 활용해 LLM에 질문하여 답변을 받아 정확성을 확인한다.

### 시도 1

#### 사전준비 - PDF 파일 기반의 벡터 데이터 베이스 구축

```python
import fitz  # PyMuPDF

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_text_splitters import LatexTextSplitter


# 사전준비: PDF 파일 기반의 벡터 데이터 베이스 구축
# => PDF의 테이블 정보를 정제하여, 마크다운으로 변환하는 방법


# PDF의 페이지별로 텍스트 및 테이블 추출
def extract_pdf(pdf_document):
    data = []

    for page_num in range(num_pages):
        page = pdf_document.load_page(page_num)

        # 텍스트 추출
        text_blocks = page.get_text("dict")["blocks"]
        for block in text_blocks:
            if block["type"] == 0:  # 텍스트 블록
                bbox = block[
                    "bbox"
                ]  # 텍스트나 다른 요소들의 위치와 크기를 정의하는 사각형 영역
                text = block["lines"]
                # 블록 별 라인의 글자(텍스트) 영역을 한 줄로 합치기
                text_content = " ".join(line["spans"][0]["text"] for line in text)

                data.append(
                    {
                        "type": "text",
                        "content": text_content,
                        "bbox": bbox,
                        "page_num": page_num,
                    }
                )

        # 테이블 추출
        tables = page.find_tables()
        if len(tables.tables) > 0:
            for table in tables.tables:
                # table_content = table.to_pandas().to_string(index=False)
                table_content = table.to_pandas().to_markdown(index=False)
                # print(dataes)
                data.append(
                    {
                        "type": "table",
                        "content": "\n" + table_content + "\n",
                        "bbox": table.bbox,
                        "page_num": page_num,
                    }
                )
    return data


# 추출된 테이블의 위치에 맞게 정렬
def sort_data(data):
    sorted_data = sorted(
        data, key=lambda x: (x["page_num"], x["bbox"][1], x["bbox"][0])
    )
    return sorted_data


# 1. PDF 파일 읽기
pdf_document = fitz.open("../BXCM_Developer_Guide_Development_Standard.pdf")
num_pages = pdf_document.page_count

# 2. PDF의 페이지별로 텍스트 및 테이블 정보 각각 추출 (테이블은 Markdown 형식으로 반환하여 저장)
data = extract_pdf(pdf_document)

# 3. 추출된 정보를 텍스트와 테이블이 원본 데이터처럼 순서 맞추기
sorted_raw_data = sort_data(data)
# print(sorted_data)  # keys: type, content, bbox, page_num
# source_data: content만 저장
source_data = [d["content"] for d in sorted_raw_data]
# source_meta_data: 메타 데이터로 활용하기 위해 page_num과 type만 저장
source_meta_data = [{"page": d["page_num"], "type": d["type"]} for d in sorted_raw_data]

# 4. 문장 나누기
# splitter 초기화
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000, chunk_overlap=200, separators=["\n"]
)
# 추출한 정보 Document 타입으로 변환
docs = text_splitter.create_documents(source_data, source_meta_data)
# Document 나누기
splitted_documents = text_splitter.split_documents(docs)

# 5. 분할된 문장을 임베딩하여 데이터 베이스 저장
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(splitted_documents, embeddings)
db.save_local("faiss_table_index")  # Replace faiss_index with name of your choice

print("데이터베이스 생성이 완료되었습니다.")
```

[Pdf파일에서 테이블 내용 추출하기](https://forum.bwg.co.kr/t/pdf/191) 에서 기재한 방법을 이용하여 vector database를 구축하였다. -> `faiss_table_index`

이것을 가지고 검색 및 프롬프트 구축을 통해 테스트를 진행해보았다.

#### 답변 생성 테스트

> **결과**

- 검색된 관련 문서
  
  ```
  # doc =>  page_content='[표 2.2]\xa0온라인 어플리케이션 용어' metadata={'page': 20, 'type': 'text'}
  # doc =>  page_content='2.2.1 온라인 어플리케이션' metadata={'page': 20, 'type': 'text'}
  # doc =>  page_content='[표 2.3]\xa0배치 어플리케이션 용어' metadata={'page': 21, 'type': 'text'}
  # doc =>  page_content='2.4 온라인 어플리케이션 명명규칙' metadata={'page': 24, 'type': 'text'}
  ```

- LLM 답변
  
  ```
  문장에 따르면, 온라인 어플리케이션 용어에 대한 설명은 [표 2.2]에 나와 있을 것으로 보입니다.
  그러나 주어진 문장에는 [표 2.2]의 구체적인 내용이 포함되어 있지 않기 때문에, 온라인 어플리케이션 용어에 대한 자세한 설명을 제공할 수 없습니다.
  [표 2.2]를 직접 참조하여 필요한 정보를 확인하는 것이 좋습니다.
  ```

#### 📌 결과 분석

질문에 대한 답변을 제대로 처리하지 못한다. 

원인은 다음과 같이 볼 수 있다.

- 임베딩된 chunk의 단위가 너무 작다.

- table의 내용이 핵심 내용인데, table의 정보와 테이블이 무엇을 의미하는지 테이블을 설명하는 부분이 서로 같은 chunk내에 존재하지 않으므로, 라벨링(표의 이름)만 관련 데이터로 추출되고, 핵심 내용은 table은 관련 데이터로 추출되지 못한다.

---

## 시도 2

`시도 1`에서 청크를 잘못 나누어 문제가 되었던 것으로 확인된다. 

PDF를 로딩해올 때 사용했던 라이브러리인 `PyMuPDFLoader`의 공식문서에서 다른 방법은 없을까 보던 중 [PyMuPDF, LLM&RAG](https://pymupdf.readthedocs.io/en/latest/rag.html)문서를 보게되었다. 보니 PDF를 Markdown으로 변환하는 라이브러리가 소개되어있었다.

[PyMuPDF4LLM Document](https://pymupdf4llm.readthedocs.io/en/latest/index.html)를 참고해서 아래와 같이 진행해보았다.

- PDF 파일 로딩하여, 마크다운으로 변환: `pymupdf4llm` 이용

- Markdown Header 기준으로 자르기: `MarkdownHeaderTextSplitter` 이용

- 분할된 문장을 임베딩하여 데이터 베이스 저장: `FAISS` DB 이용

#### 사전준비 - PDF 파일 기반의 벡터 데이터 베이스 구축

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import pymupdf4llm
from langchain.text_splitter import MarkdownHeaderTextSplitter

# 사전준비: PDF 파일 기반의 벡터 데이터 베이스 구축
# => PDF를 마크다운으로 변환하여, 마크다운 헤더 기준으로 청크를 구분하는 방법

# 1. PDF 파일 로딩하여, 마크다운으로 변환
md_text = pymupdf4llm.to_markdown("../test_pdfs/sample_test.pdf")
# print(md_text)

# 2.문장 나누기
# Header 기준으로 나누기
# Header 1 나누고 -> Header 2 나누고 -> ...
# 상대적으로 크기가 나눠지기 때문에 구조를 정확히 잡기 위해서는 Header6까지 잡아주는 것이 안전
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
    ("######", "Header 6"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False # 분할된 청크 내의 content에서 헤더 제거 여부
)
splitted_documents = markdown_splitter.split_text(md_text)
# print(len(md_header_splits))

# 3. 분할된 문장을 임베딩하여 데이터 베이스 저장
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(splitted_documents, embeddings)
db.save_local("faiss_index_test-md") # Replace faiss_index with name of your choice

print("데이터베이스 생성이 완료되었습니다.")
```

추출된 markdown은 다음과 같다. 매우 깔끔하게 잘 추출되는 것을 확인할 수 있다.

<details>
<summary>추출된 markdown</summary>

```markdown
**2.2.1 온라인 어플리케이션**

[표 2.2] 온라인 어플리케이션 용어

|구분|설명|
|---|---|
|어플리케이션|•사용자의 요청을 처리하기 위한 프로그램의 집합으로 데이터 액세스와 처리를 위해 작성된 SQL, 비즈니스 로직 구현을 위한 클래스, 설정파일들을 포함하여 구성된다. •어플리케이션은 Eclipse의 BX 클라우드 프로젝트로 작성된다. •본 프로젝트의 업무 개발자는 별도로 어플리케이션을 생성하지 않고 이미 작성되어 있는 Git Repositor에 제공된 어플리케이션을 Clone하여 사용한다. •외부에서 식별하기 위해 시스템 내에서 유일한 이름을 가진다.|
|컨트롤러 (Controller)|•Rest Client(사용자 화면, 대내외채널 등 외부에서 호출된 사용자 요청)의 요청에 대 응되는 클래스이다. •HTTP 클라이언트 요청을 처리(HTTP 의존적인 영역을 처리)힌다. •트랜잭션 처리 단위의 기준이 된다. •비즈니스 로직을 처리하기 위해 Service를 호출한다. •Component와 DBIO는 직접 처리하지 는다.|
|서비스(Service)|•Controller와 1:1 로 매핑되는 클래스이다 •Controller에서 호출되는 단위의 기능을 Service의 메소드로 구현한다.|
|컴포넌트 (Component)|•업무 기능 중 공통적으로 사용 할 수 있는 기능을 추상화하여 제공하는 클래스이다. •재사용이 가능한 업무 단위라고 볼 수 있다.|
|DBIO|•데이터 액세스를 위해 작성하는 개발 리소스로 자바 Interface와 SQL Mapper(.dbio)파일로 구성된다. •개발도구에서 DBIO를 생성하여 개발한다. •SQL은 DBIO에서 SQL ID를 추가하여 작성한다.|
|SQL ID|•DBIO에서 생성되는 쿼리를 나타내는 식별자이다. DBIO 인터페이스의 메소드 이름 이 된다.|
|IO|•메시지(JSON, 고정길이 데이터, XML 등) 형태로 변환할 수 있는 기능을 제공하는 프레임워크 표준 데이터 전달 객체로 Controller 요청/응답, 대내외계 인터페이스, 비즈니스 호출 등에 사용된다. •모든 DTO(Data Transfer Object)는 IO로 관리된다.|
|일반 클래스|•순수 자바 클래스로 사용자가 자유롭게 작성이 가능하다.|
|JUnit 테스트 케 이스|JUnit을 사용해 업무 소스를 테스트 할 수 있는 테스트 파일.|


**2 명명규칙** **14**


-----

**2.2.2 배치 어플리케이션**

[표 2.3] 배치 어플리케이션 용


|구분|설명|
|---|---|
|어플리케이션|배치 작업을 위한 프로그램의 집합으로 배치작업 설정파일, 데이터 액세스와 처리를 위 해 작성된 SQL, 비즈니스 로직 구현을 위한 클래스들을 포함하여 구성된다.|
|Job|배치 업무를 처리하기 위한 플로우로 구성된 배치작업의 실행 및 재실행 단위로 플로우 를 기술하는 XML 설정으로 작성된다.|
|Step|Job을 구성하는 업무 구현의 최소단위이다.|
|Job Configuration 설정 파일|Job 과 Step 의 옵션 및 실행 순서를 설정하는 파일이다.|


**2.2.3 기타**

[표 2.4] 기타 용어

|구분|설명|
|---|---|
|거래ID|특정 거래를 처리하는 Rest API 식별자|
|작업 ID|배치 Job을 구분하는 식별자|
|로그 파일|거래 로그 혹은 배치 수행 로그를 저장하는 파일|



# 2.3 공통 명명규칙

모든 명명규칙은 다음의 공통 사항을 포함한다.

**2.3.1 제약사항**

(1) 메타에서 제공하는 메타 데이터를 제외한 나머지 항목은 한글 발음으로 된 영문자 사용을 제한한다. 예)직업>Jikup

(2) 영문자를 제외한 기호, 특수 문자 등은 사용을 제한한다. 예) _, $

(3) 테이블이름을 사용하는 경우 예외로 한다

(4) 표준단어 영문약어 조합으로 각 컴포넌트를 명명한다.

**2.3.2 표준단어**

(1) 표준단어를 다음의 카멜 표기법으로 변환하여 사용한다.

(2) 카멜 표기법 변환 규칙: _(언더바)로 구분된 단어에서..

**2 명명규칙** **15**
```

</details>


수행 결과 청크는 아래와 같이 구분되었다.

![](uploads/03_PDF%20내%20Table%20추출하여%20RAG%20APP%20만들기/2024-06-25-10-25-16-image.png)

> **💡추가 정보**

현재 테스트 데이터로 쓴 문서의 경우, 처음부터가 아닌 중간에 header가 짤린 부분 페이지를 사용하여서 계층구조가 제대로 잡히지 않은 것으로 보인다.

하지만 `MarkdownHeaderTextSplitter`를 통해 구분할 경우, `meta`에 계층 정보를 잘 포함하고 있다.

```markdown
# Foo
## Bar
Hi this is Jim
Hi this is Joe
### Boo 
Hi this is Lance

## Baz
Hi this is Molly
```

위와 같은 계층이 존재할 경우 아래와 같이 청크로 구분된다.

```python
[Document(page_content='Hi this is Jim  \nHi this is Joe', metadata={'Header 1': 'Foo', 'Header 2': 'Bar'}),
 Document(page_content='Hi this is Lance', metadata={'Header 1': 'Foo', 'Header 2': 'Bar', 'Header 3': 'Boo'}),
 Document(page_content='Hi this is Molly', metadata={'Header 1': 'Foo', 'Header 2': 'Baz'})]
```



#### 답변 생성 테스트

> **결과**

- 검색된 관련 문서
  
  ```
  # doc =>  page_content='**2.2.1 온라인 어플리케이션**  \n[표 2.2] 온라인 어플리케이션 용어  \n|구분|설명|\n|---|---|\n|어플리케이션|•사용자의 요청을 처리하기 위한 프로그램의 집합으로 데이터 액세스와 처리를 위해 작성된 SQL, 비즈니스 로직 구현을 위한 클래스, 설정파일들을 포함하여 구성된다. •어플리케이션은 Eclipse의 BX 클라우드 프로젝트로 작성된다. •본 프로젝트의 업무 개발자는 별도로 어플리케이션을 생성하지 않고 이미 작성되어 있는 Git Repositor에 제공된 어플리케이션을 Clone하여 사용한다. •외부에서 식별하기 위해 시스템 내에서 유일한 이름을 가진다.|\n|컨트롤러 (Controller)|•Rest Client(사용자 화면, 대내외채널 등 외부에서 호출된 사용자 요청)의 요청에 대 응되는 클래스이다. •HTTP 클라이언트 요청을 처리(HTTP 의존적인 영역을 처리)힌다. •트랜잭션 처리 단위의 기준이 된다. •비즈니스 로직을 처리하기 위해 Service를 호출한다. •Component와 DBIO는 직접 처리하지 는다.|\n|서비스(Service)|•Controller와 1:1 로 매핑되는 클래스이다 •Controller에서 호출되는 단위의 기능을 Service의 메소드로 구현한다.|\n|컴포넌트 (Component)|•업무 기능 중 공통적으로 사용 할 수 있는 기능을 추상화하여 제공하는 클래스이다. •재사용이 가능한 업무 단위라고 볼 수 있다.|\n|DBIO|•데이터 액세스를 위해 작성하는 개발 리소스로 자바 Interface와 SQL Mapper(.dbio)파일로 구성된다. •개발도구에서 DBIO를 생성하여 개발한다. •SQL은 DBIO에서 SQL ID를 추가하여 작성한다.|\n|SQL ID|•DBIO에서 생성되는 쿼리를 나타내는 식별자이다. DBIO 인터페이스의 메소드 이름 이 된다.|\n|IO|•메시지(JSON, 고정길이 데이터, XML 등) 형태로 변환할 수 있는 기능을 제공하는 프레임워크 표준 데이터 전달 객체로 Controller 요청/응답, 대내외계 인터페이스, 비즈니스 호출 등에 사용된다. •모든 DTO(Data Transfer Object)는 IO로 관리된다.|\n|일반 클래스|•순수 자바 클래스로 사용자가 자유롭게 작성이 가능하다.|\n|JUnit 테스트 케 이스|JUnit을 사용해 업무 소스를 테스트 할 수 있는 테스트 파일.|  \n**2 명명규칙** **14**  \n-----  \n**2.2.2 배치 어플리케이션**  \n[표 2.3] 배치 어플리케이션 용어  \n|구분|설명|\n|---|---|\n|어플리케이션|배치 작업을 위한 프로그램의 집합으로 배치작업 설정파일, 데이터 액세스와 처리를 위 해 작성된 SQL, 비즈니스 로직 구현을 위한 클래스들을 포함하여 구성된다.|\n|Job|배치 업무를 처리하기 위한 플로우로 구성된 배치작업의 실행 및 재실행 단위로 플로우 를 기술하는 XML 설정으로 작성된다.|\n|Step|Job을 구성하는 업무 구현의 최소단위이다.|\n|Job Configuration 설정 파일|Job 과 Step 의 옵션 및 실행 순서를 설정하는 파일이다.|  \n**2.2.3 기타**  \n[표 2.4] 기타 용어  \n|구분|설명|\n|---|---|\n|거래ID|특정 거래를 처리하는 Rest API 식별자|\n|작업 ID|배치 Job을 구분하는 식별자|\n|로그 파일|거래 로그 혹은 배치 수행 로그를 저장하는 파일|'
  # doc =>  page_content='# 2.3 공통 명명규칙  \n모든 명명규칙은 다음의 공통 사항을 포함한다.  \n**2.3.1 제약사항**  \n(1) 메타에서 제공하는 메타 데이터를 제외한 나머지 항목은 한글 발음으로 된 영문자 사용을 제한한다. 예)직업>Jikup  \n(2) 영문자를 제외한 기호, 특수 문자 등은 사용을 제한한다. 예) _, $  \n(3) 테이블이름을 사용하는 경우 예외로 한다  \n(4) 표준단어 영문약어 조합으로 각 컴포넌트를 명명한다.  \n**2.3.2 표준단어**  \n(1) 표준단어를 다음의 카멜 표기법으로 변환하여 사용한다.  \n(2) 카멜 표기법 변환 규칙: _(언더바)로 구분된 단어에서..  \n**2 명명규칙** **15**  \n-----' metadata={'Header 1': '2.3 공통 명명규칙'}
  ```

- LLM 답변
  
  ```markdown
  온라인 어플리케이션 용어는 다음과 같이 설명됩니다:
  
  1. **어플리케이션**:
     - 사용자의 요청을 처리하기 위한 프로그램의 집합으로, 데이터 액세스와 처리를 위해 작성된 SQL, 비즈니스 로직 구현을 위한 클래스, 설정파일들을 포함하여 구성됩니다.
     - 어플리케이션은 Eclipse의 BX 클라우드 프로젝트로 작성됩니다.
     - 본 프로젝트의 업무 개발자는 별도로 어플리케이션을 생성하지 않고 이미 작성되어 있는 Git Repositor에 제공된 어플리케이션을 Clone하여 사용합니다.
     - 외부에서 식별하기 위해 시스템 내에서 유일한 이름을 가집니다.
  
  2. **컨트롤러 (Controller)**:
     - Rest Client(사용자 화면, 대내외채널 등 외부에서 호출된 사용자 요청)의 요청에 대응되는 클래스입니다.
     - HTTP 클라이언트 요청을 처리(HTTP 의존적인 영역을 처리)합니다.
     - 트랜잭션 처리 단위의 기준이 됩니다.
     - 비즈니스 로직을 처리하기 위해 Service를 호출합니다.
     - Component와 DBIO는 직접 처리하지 않습니다.
  
  3. **서비스 (Service)**:
     - Controller와 1:1로 매핑되는 클래스입니다.
     - Controller에서 호출되는 단위의 기능을 Service의 메소드로 구현합니다.
  
  4. **컴포넌트 (Component)**:
     - 업무 기능 중 공통적으로 사용할 수 있는 기능을 추상화하여 제공하는 클래스입니다.
     - 재사용이 가능한 업무 단위라고 볼 수 있습니다.
  
  5. **DBIO**:
     - 데이터 액세스를 위해 작성하는 개발 리소스로 자바 Interface와 SQL Mapper(.dbio) 파일로 구성됩니다.
     - 개발도구에서 DBIO를 생성하여 개발합니다.
     - SQL은 DBIO에서 SQL ID를 추가하여 작성합니다.
  
  6. **SQL ID**:
     - DBIO에서 생성되는 쿼리를 나타내는 식별자입니다. DBIO 인터페이스의 메소드 이름이 됩니다.
  
  7. **IO**:
     - 메시지(JSON, 고정길이 데이터, XML 등) 형태로 변환할 수 있는 기능을 제공하는 프레임워크 표준 데이터 전달 객체로, Controller 요청/응답, 대내외계 인터페이스, 비즈니스 호출 등에 사용됩니다.
     - 모든 DTO(Data Transfer Object)는 IO로 관리됩니다.
  
  8. **일반 클래스**:
     - 순수 자바 클래스로 사용자가 자유롭게 작성이 가능합니다.
  
  9. **JUnit 테스트 케이스**:
     - JUnit을 사용해 업무 소스를 테스트할 수 있는 테스트 파일입니다.
  
  ```

#### 📌 결과 분석

이번엔 완벽하게 구조를 파악하여 제대로 된 답변을 주는 것을 확인할 수 있다.

우선 청크단위가 Header단위로 나뉘기 때문에 맥락이 끊기지 않는다는 것이 큰 장점이다.

위 기재된 테스트코드의 경우는 병합된 셀이 존재하지 않는 테이블이었지만, 병합된 셀이 존재하는 테이블의 경우도 추출이 잘 되는 것을 확인했다. 

<img src="uploads/03_PDF%20내%20Table%20추출하여%20RAG%20APP%20만들기/2024-06-25-10-47-28-image.png" title="" alt="" data-align="left">

위와 같이 `사용할 수 있는 값` 셀이 3줄이 병합되어있는 그러한 경우에는 아래와 같은 마크다운 표로 변환해주는 것을 확인했다.

|속성|설명|
|---|---|
|키|database.datasource.mapping|
|다중 값 여부|예|
|사용할 수 있는 값|MainDS,Bxm Connection|
||MainDS,Oracle Connection,oracle|
||MainDS,MySql Connection,mysql|


---

추후 작업으로는 아래의 테스트를 이어서 해볼 예정이다.

- `meta`에 대한 계층 테스트: 계층 구조를 이해하여 답변을 생성하는지

- `meta` 정보를 이용하여 어플리케이션에서 참고문서로 어떻게 활용하면 좋을 지 생각해보기

- 클래스화: PDF 로딩 -> md로 변환 -> header 단위 split -> 임베딩 -> 벡터스토어 저장 과정
