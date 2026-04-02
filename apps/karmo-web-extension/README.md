# KarmoWebExtension (Chrome MV3)

로컬 앱(`chat-overlay` 등)과 브라우저 방송·시청 페이지를 잇는 **KarmoWebExtension**입니다.

## 개발용 로드

1. Chrome `chrome://extensions` → **개발자 모드**
2. **압축해제된 확장 프로그램을 로드합니다** → 이 폴더(`karmo-web-extension`) 선택

기존에 `stream-overlay-extension` 등으로 로드했다면 제거 후 이 폴더를 다시 로드하세요.

## 다음 작업 예시

1. 치지직 라이브 시청 페이지에서 채팅 DOM 구조 확인
2. `content.js`의 `extractChatRows` 구현
3. **chat-overlay(Tauri)** 가 떠 있으면 `127.0.0.1:17376` 에서 `POST /ingest` 를 받고 오버레이로 표시함

## 설정

- 기본 수신 URL: `http://127.0.0.1:17376/ingest`
- `chrome.storage.sync`의 `ingestUrl`로 변경 가능

## 주의

각 방송 플랫폼 약관·정책은 직접 확인하세요.
