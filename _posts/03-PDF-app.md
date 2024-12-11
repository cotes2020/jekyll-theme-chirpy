[PymuPdf 라이브러리를 이용한 PDF의 Markdown 변환](03_PDF-Table-추출)을 통해 PDF에 있는 Table을 깔끔하게 반환해오는 작업을 진행했었습니다.

이를 활용해서 front를 붙이며 어플리케이션을 개발해보고,

보완할 수 있는 사항들을 적용해보았습니다.

---

어플리케이션 모습입니다.

- 첫화면

![image-20240626183046325](uploads/03-PDF-app/image-20240626183046325.png)

- 파일 업로드 후 업로드 완료

기존에도 테스트 진행했던 PDF 파일을 넣어줍니다. 임베딩 후 완료 메시지를 보여줍니다.

![image-20240626183117383](uploads/03-PDF-app/image-20240626183117383.png)

- 질문

이전과 같은 질문을 해볼까요?

![1719394343301](image/03-PDF-app/1719394343301.png)

역시 모든 항목에 대해 답을 잘 주는군요. 

참고 목록에 있는 원문을 눌러볼까요?

- 참고부분 확인

![1719394388304](image/03-PDF-app/1719394388304.png)

표가 좀 깨지긴 했지만 꽤나 깔끔하게 보여줍니다 ㅎㅎ


---

구현한 코드는 아래와 같습니다.

> app.py

```python
import re
import chainlit as cl
from langchain.chains import ConversationalRetrievalChain
import pymupdf4llm
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

def convert_header(span, page):
    """
    Determines the markdown header level based on the text prefix in the span.
    `*.*.*` 형식을 Header6로 지정한다.

    Args:
    span (dict): Dictionary containing at least a 'text' key with the text of the document span.

    Returns:
    str: A Markdown header level string ("## " for second-level, "### " for third-level, or "" otherwise).
    """
    text = span.get('text', '')
    size = span.get('size', 10)

    # 제목 헤더 없애기
    if size > 35:
        return ''

    # 너무 작은 경우 header 주지 않기
    if size <= 12:
        # print(span)
        return ''

    # Header 3단계까지 재구성
    # Check if text starts with "1. ", "2. ", etc.
    if re.match(r'^\d+\.\s', text):
        return '# '
    # Check if text starts with "1.1 ", "1.2 ", etc.
    elif re.match(r'^\d+\.\d+\s', text):
        return '## '
    # Check if text starts with "1.1.1 ", "1.2.1 ", etc.
    if re.match(r'^\d+\.\d+.\d+\s', text):
        return '### '
    else:
        return ''
  
# 채팅 화면 만들기
@cl.on_chat_start  # ← 채팅이 시작될 때 실행할 함수를 정의
async def on_chat_start():

    # 채팅 시작 시 파일 업로드 기능 구현
    files = None

    while files == None:
        files = await cl.AskFileMessage(
            content="PDF 파일을 업로드해주세요!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    # 1. 업로드된 PDF 로드하여 마크다운으로 변환
    # + 커스터마이징 header 변환 함수 추가
    md_text = pymupdf4llm.to_markdown(file.path,
                                      hdr_info=convert_header)
    print(md_text)

    msg = cl.Message(content=f"Processing `{file.name}`...", disable_feedback=True)
    await msg.send()

    # 2. Header 기준으로 청크 나누기
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    # splitter 초기화
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,  # 분할된 청크 내의 content에서 헤더 제거 여부
    )
    # markdown 파일을 header 기준으로 청크 분리
    splitted_documents = markdown_splitter.split_text(md_text)

    # 3. 분할된 청크를 임베딩하여 데이터 베이스 생성
    embeddings = OpenAIEmbeddings()
    db = await cl.make_async(FAISS.from_documents)(splitted_documents, embeddings)

    # 4. 메모리 생성하여 vector store를 메모리로 저장
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # 5. FAISS voctor store memory를 이용한 chain 생성
    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True),
        chain_type="stuff",
        retriever=db.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    msg.content = f"`{file.name}` 파일 업로드가 완료되었습니다!🤗\n질문을 입력하세요!✏️"
    await msg.update()

    # 6. session에 vector store 저장 내용을 저장
    cl.user_session.set("chain", chain)


# 8. 업로드된 파일을 기반으로 답변 받기
@cl.on_message  # ← 메시지를 보낼 때 실행할 함수를 정의
async def on_message(message: cl.Message):
    # 7. session에 저장된 vector store 내용 획득
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()

    # 8. 입력된 메시지를 통해 chain 수행
    # - vecter store에서의 질문과 관련된 문서 추출
    # - LLM에 질문하여 답변 받기
    res = await chain.acall(message.content, callbacks=[cb])

    # answer: LLM의 답변
    answer = res["answer"]
    # source_documents: vecter store에서의 질문과 관련된 문서
    source_documents = res["source_documents"]  # type: List[Document]
    print(source_documents)

    # 출처 페이지 사이드 바 구성
    text_elements = []  # type: List[cl.Text]

    # 9. 관련 문서가 존재하는 경우 답변 생성
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):

            # 출처 페이지 구성
            text_elements.append(
                cl.Text(
                    content=source_doc.page_content, name=' > '.join(list(source_doc.metadata.values())), display="side"
                )
            )

        source_names = [text_el.name for text_el in text_elements]

        if source_names:

            answer += f"\n\n **참고**"
            for source in source_names:
                answer += f"\n- {source}"
        else:
            answer += "\n문서에 질문에 대한 정보가 존재하지 않습니다."

    # 10. 답변에 LLM의 답변과 참고 문서에 대한 정보를 같이 반환
    await cl.Message(content=answer, elements=text_elements).send()

```

## 주요 작업

### Front 구성

`chainlit`을 이용하였습니다.

- `on_chat_start()`
  - 처음 PDF 파일을 업로드를 요청합니다.
  - Indexing
    - PDF 파일 로딩하여 markdown변환합니다.
    - markdown은 Header 단위로 청크 구분하여 임베딩합니다.
    - vector DB에 저장합니다.
  - Chain 생성
    - vector DB를 retriever로 변환합니다.
    - 대화내용을 메모리로 저장합니다.
    - 모델을 할당합니다.
  - session에 chain 저장
- `on_message()`
  - 메시지를 입력받으면, 답변을 생성하여 사용자에게 보여줍니다.
  - session으로부터 chain 호출
  - 질문에 대한 답변 반환
    - 답변에서 `source_documents`를 추출하여 사이드바에 `참고` 로 보여줍니다.

### PDF to Markdown

기존에는 순수하게 PDF 파일을 바로 Markdown으로 반환했는데, 이번에는 한가지 변환과정을 거쳤습니다.

`pymupdf4llm.to_markdown`에 `hdr_info` 파라미터의 값을 추가했습니다.

> [!TIP]
>
> 각 PDF 파일에 최적화되도록 `hdf_info`를 통해 header 기준을 정의합니다.

기존 테스트를 진행하였던 `BXCM_Developer_Guide_Development_Standard.pdf` 파일을 가장 작은 소제목 기준으로 chunk를 나누고 싶었습니다.

그런데 markdown으로 변환하면 가장 작은 소제목은 header가 아닌 그냥 값으로 인식되었습니다.

예를들어 아래와 같은 부분이 있습니다.

![image-20240626181410316](uploads/03-PDF-app/image-20240626181410316.png)

저는 `2.4.1 어플리케이션` 부분이 하나의 청크로 묶으면 좋겠으나, markdown은 아래와 같이 변환됩니다.

```
**2.4.1 어플리케이션**  \n어플리케이션 단위는 관리 주체가 다르고, 어
리케이션 간 기능 호출을 최소화 할 수 있는 단위로 구분 한 ... (생략)
```

그렇기 때문에 청크 단위는 아래와 같이 `2.4 온라인 어플리케이션 명명규칙` 단위로 묶입니다.

```
'## 2.4 온라인 어플리케이션 명명규칙  \n**2.4.1 어플리케이션**  \n어플리케이션 단위는 관리 주체가 다르고, 어
리케이션 간 기능 호출을 최소화 할 수 있는 단위로 구분 한다. 어플리\n케이션은 Studio의 \'BX 클라우드 프로젝트\' 메뉴를 통해 생성한다.  \n[표 2.8] 어플리리
이션 명명  \n|구분|설명|\n|---|---|\n|구성|L1코드 ... (생략)
```

그래서 **`숫자`.`숫자`.`숫자` 패턴**을 가진 경우 Header3로 인식되도록 함수를 생성했습니다.

```python
def convert_header(span, page):
    text = span.get('text', '')
    size = span.get('size', 10)

    # 제목 헤더 없애기
    if size > 35:
        return ''

    # 너무 작은 경우 header 주지 않기
    if size <= 12:
        # print(span)
        return ''

    # Header 3단계까지 재구성
    # Check if text starts with "1. ", "2. ", etc.
    if re.match(r'^\d+\.\s', text):
        return '# '
    # Check if text starts with "1.1 ", "1.2 ", etc.
    elif re.match(r'^\d+\.\d+\s', text):
        return '## '
    # Check if text starts with "1.1.1 ", "1.2.1 ", etc.
    if re.match(r'^\d+\.\d+.\d+\s', text):
        return '### '
    else:
        return ''
```

또한 제목인 `BX Framework`의 경우 의미는 없지만 폰트가 가장 커서 header1으로 잡혀 무의미하기에 제목은 헤더에서 제거했습니다.

또한 너무 작은 글씨는 header를 주지 않는 설정을 했습니다.

위와 같은 전처리를 통해 아래와 같이 markdown이 변환되었고, 청크를 header3단위까지 잡을 수 있었습니다.

![image-20240626182653299](uploads/03-PDF-app/image-20240626182653299.png)

이를 통해 보다 구체적인 정보를 얻을 수 있고, `참고`에서도 참고 원문을 명확히 찾을 수 있었습니다.

---

PDF 파일마다 `hdr_info`를 통해 커스터마이징을 해주어야 명확한 답을 얻을 수 있다는 것이 한계이군요...
