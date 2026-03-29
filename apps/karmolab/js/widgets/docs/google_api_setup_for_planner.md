# KarmoLab Planner - Google API 연결 가이드

이 문서는 KarmoLab Planner 타 구글 계정으로의 확장이나 환경 재설정 시 필요한 **Google API 및 OAuth 설정 과정**을 기록합니다.

## 1. Google Cloud Console 세팅
1. [Google Cloud Console](https://console.cloud.google.com/)에 접속하여 로그인합니다.
2. 좌측 상단 프로젝트 선택 드롭다운에서 **[새 프로젝트]**를 생성합니다. (예: `karmolab-planner`)

## 2. API 활성화 (Enable APIs)
1. 좌측 메뉴에서 **API 및 서비스 > 라이브러리**로 이동합니다.
2. 다음 두 가지 API를 검색하여 각각 **[사용(Enable)]**을 클릭합니다:
   - **Google Calendar API** (일정 관리용)
   - **Google Tasks API** (TODO 카드 관리용)

## 3. OAuth 동의 화면 구성
1. 좌측 메뉴에서 **API 및 서비스 > OAuth 동의 화면**으로 이동합니다.
2. User Type을 **외부(External)**로 선택하고 만들기를 클릭합니다.
   *(본인만 사용할 경우에도 외부로 두고 앱 게시 상태를 '테스트 중'으로 사용하거나 추가 비용 없이 프로덕션으로 전환 가능)*
3. **앱 이름**, **사용자 지원 이메일**, **개발자 연락처 정보**를 입력합니다.
4. (선택) 하단에 '테스트 사용자' 부분에 본인의 구글 이메일을 추가해야 테스트 단계에서 로그인이 가능합니다.

## 4. 사용자 인증 정보 (Client ID) 발급
1. 좌측 메뉴에서 **API 및 서비스 > 사용자 인증 정보(Credentials)**로 이동합니다.
2. 상단 **[+ 사용자 인증 정보 만들기]** > **[OAuth 클라이언트 ID]**를 클릭합니다.
3. 애플리케이션 등급(Application Type)을 **[웹 애플리케이션]**으로 선택합니다.
4. **승인된 자바스크립트 원본(Authorized JavaScript origins)** 항목에 웹 앱이 실행될 주소들을 정확히 추가합니다. (마지막의 `/`는 제외)
   - `http://localhost:4000` (Jekyll 로컬 테스트용)
   - `http://localhost:5173` (Vite 로컬 개발 서버용)
   - `https://mascari4615.github.io` (실제 블로그 배포 주소)
   - (*주의*: `승인된 리디렉션 URI` 칸이 아닙니다. 자바스크립트 원본 칸에 넣어야 합니다!)
5. 생성 후 팝업에 나타나는 **클라이언트 ID (Client ID)** 문자열을 복사합니다. (비밀번호 역할인 Client Secret은 프론트엔드 환경에서 사용하지 않습니다)

## 5. 앱 환경 변수에 적용
Vite React 프로젝트 디렉토리 내부의 `.env` (또는 환경 변수 설정 파일)에 발급받은 클라이언트 ID를 기입합니다.
```env
VITE_GOOGLE_CLIENT_ID="1234567890-abcdefg...apps.googleusercontent.com"
```

## 참고사항
이 API 연동 방식은 **Implicit Grant Flow** (또는 GIS 라이브러리)를 따릅니다. 서버나 DB 없이 브라우저 단에서 임시 토큰(액세스 토큰)을 발급받아 사용하는 완전히 종단간(클라이언트 - 구글 서버) 안전한 인증 방식입니다.
