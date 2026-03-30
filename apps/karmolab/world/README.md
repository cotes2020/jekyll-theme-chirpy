## KarmoWorld (세계관 SSoT)

이 디렉터리는 KarmoLab 세계관 데이터를 **종류(kind)**와 **쓰임세(purpose)**로 나눠 관리하기 위한 단일 출처입니다.

### 구조

- `entities/`: 인물·세력·장소 등 **개체(entity)** 정의(코어 사실, 관계 포인터 위주)
- `artifacts/`: 아이템·문헌·부적 등 **아티팩트(artifact)** 정의(코어 사실, 관계 포인터 위주)
- `bindings/`: 도구별 **투영(binding)** 레이어
  - `bindings/imagegen/`: 이미지 생성 프리셋에 필요한 프롬프트 조각 및 매핑
  - `bindings/chatbot/`: 챗봇 캐릭터 카드 필드 및 매핑
- `wiki/`: 사람이 읽고 쓰는 장문 위키(Markdown)
  - `wiki/entities/...`
  - `wiki/artifacts/...`

### 원칙

- 코어(entities/artifacts)는 길어져도 “세계 안에서의 사실”을 중심으로 유지합니다.
- 도구별 문구/프롬프트는 bindings로 분리해, 같은 개체를 여러 도구가 각자 필요한 형태로 소비합니다.

