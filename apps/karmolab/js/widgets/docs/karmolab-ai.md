# KarmoLabAI (`karmolab-ai`) 사용 가이드

> **원본 경로:** 레포 루트 `packages/karmolab-ai/` — npm 패키지 이름 `karmolab-ai`  
> **역할:** Google **AI Studio**(Generative Language API)와 **Vertex AI**를 함께 쓸 때, 모델 ID·REST URL·문서 링크·기본값을 **한곳(SSOT)**에서 맞춥니다. `fetch`·API 키 저장·DOM은 포함하지 않습니다.

---

## 무엇이 들어 있나

| 구분 | 설명 |
|------|------|
| **모델 카탈로그** | 텍스트 Gemini(`gemini`), 이미지용 Nano Banana(`geminiImage`), Imagen(`imagen`) 목록과 표시 이름 |
| **기본 모델 ID** | `DEFAULT_TEXT_MODEL_ID` — 텍스트용 기본 API 모델 ID (`gemini-2.5-flash` 등, 카탈로그와 동기) |
| **AI Studio URL** | `buildAiStudioGenerateContentUrl`, `buildAiStudioStreamGenerateContentUrl`, `buildAiStudioPredictUrl` |
| **Vertex URL** | `buildVertexPublisherModelUrl` — `generateContent` / `streamGenerateContent` / `predict` |
| **기본 리전** | `DEFAULT_VERTEX_LOCATION` (`us-central1`) |
| **문서 링크** | `DOC_URL_AI_STUDIO_API_KEY`, `DOC_URL_VERTEX_API_KEYS` |
| **Env 이름(참고)** | `ENV_GOOGLE_AI` — `GEMINI_API_KEY`, `GEMINI_MODEL` 문자열만 (값을 읽지는 않음) |
| **타입** | `GoogleGenerativeSurface`, `ModelProvider`, `ModelEntry` 등 |

소스는 TypeScript(`src/index.ts`)이고, 배포 시 `npm run build`로 `dist/`에 CommonJS가 생성됩니다.

---

## KarmoLab(브라우저)에서

- **`apps/karmolab/src/gemini.ts`** 가 `karmolab-ai`를 import합니다.
- 빌드(`apps/karmolab`에서 `npm run build`) 시 **esbuild**가 의존성을 묶어 `js/gemini.js`로보냅니다.
- 페이지 스크립트에서는 전역 **`Gemini`** 객체로 기능을 씁니다. 예:
  - **AI Studio:** `callText`, `callChat`, `callChatStream`, `callGeminiImage`, `callImagen`
  - **Vertex:** `callVertexText`, `callVertexChat`, `callVertexChatStream`, `callVertexGeminiImage`, `callVertexImagen`
- Vertex 텍스트/채팅은 사용자 설정의 **Vertex API 키**, **`ig_vertex_project_id`**, **`ig_vertex_location`**(이미지 위젯과 동일)을 사용합니다.
- **챗봇** 위젯: 사이드바 **API**에서 `Vertex AI`를 고르면 스트리밍이 Vertex 경로로 갑니다. (웹 검색은 AI Studio 전용.)

키 입력·프로필 UI는 **내 정보 → 설정**의 Gemini/Vertex 항목을 사용하세요.

---

## Node(욘봇·스크립트)에서

- **`apps/discord-bots/apps/yawnbot`** 에 `karmolab-ai`가 `file:../../../../packages/karmolab-ai` 로 연결되어 있습니다.
- 루트에서 봇 빌드할 때 `packages/karmolab-ai`가 먼저 `tsc` 됩니다 (`apps/discord-bots`의 `npm run build` / `build:yawnbot`).
- 예시:
  - `main.ts`: `import { DEFAULT_TEXT_MODEL_ID } from 'karmolab-ai'` 후 `GEMINI_MODEL` 미설정 시 기본 모델로 사용
  - `kakao-export.mjs`: `createRequire`로 `DEFAULT_TEXT_MODEL_ID` 로드

새 스크립트에서 모델 ID만 맞추고 싶다면 `MODEL_CATALOG`, `getDefaultModelId('gemini' | 'geminiImage' | 'imagen')` 를 import 해 사용하면 됩니다.

---

## 로컬에서 패키지 빌드

```bash
cd packages/karmolab-ai
npm install
npm run build
```

KarmoLab 전체 JS:

```bash
cd apps/karmolab
npm install
npm run build
```

---

## 모델 목록을 바꿀 때

1. **`packages/karmolab-ai/src/index.ts`** 의 `MODEL_CATALOG` / `isDefault` 만 수정  
2. `packages/karmolab-ai`에서 `npm run build`  
3. KarmoLab·욘봇 쪽을 각각 다시 빌드  

브라우저와 봇이 같은 ID 문자열을 쓰게 유지할 수 있습니다.

---

## 관련 문서

- Toolbox **문서 → AI 공통화** 탭: 레포 전반에서 AI 코드를 어떻게 나눌지 기획 메모  
- 사용자 **가이드** 탭: API 키 입력 위치 등 기본 사용법  

---

## 참고 링크

- [Google AI Studio API 키](https://aistudio.google.com/app/apikey)  
- [Vertex AI — API 키](https://cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys)  
- [Vertex AI REST 참고](https://cloud.google.com/vertex-ai/docs/reference/rest)
