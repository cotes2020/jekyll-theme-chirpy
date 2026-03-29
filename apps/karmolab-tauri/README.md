# KarmoLab (Tauri)

데스크톱에서 시스템 WebView로 [KarmoLab](https://mascari4615.github.io/karmolab/)을 여는 얇은 셸입니다. **앱 바이너리에는 사이트 전체를 넣지 않고**, 배포된 GitHub Pages URL을 그대로 띄웁니다. 오프라인은 **서비스 워커(Chirpy PWA)가 받아 둔 캐시**에 의존합니다(한 번 온라인으로 쓴 뒤, WebView가 캐시를 지우지 않았다면 제한적으로 동작).

## 준비물

- [Rust](https://www.rust-lang.org/learn/get-started) + [Tauri 사전 요구사항](https://v2.tauri.app/start/prerequisites/) (Windows: Visual Studio Build Tools, WebView2)
- **로컬 개발**: 저장소 루트에서 `bundle exec jekyll serve`가 돌아갈 수 있는 Ruby/Jekyll( `Gemfile` 기준)

## 명령

```bash
cd apps/karmolab-tauri
npm install
npm run dev    # Jekyll serve(4000) + 준비될 때까지 대기 후 tauri dev — 로컬 수정이 곧바로 반영
npm run dev:app  # 이미 다른 터미널에서 jekyll serve 중일 때만 Tauri만 실행
npm run dev:remote  # 개발 모드(tauri dev)인데 WebView만 배포본 URL로 — Jekyll 없이 원격 동작 테스트
npm run build    # 설치 패키지 빌드(웹은 GitHub Pages URL을 그대로 씀)
```

`tauri.conf.json`의 **`build.devUrl`** 은 **`http://127.0.0.1:4000/karmolab/`** 로 두어 개발 시 WebView2가 GitHub 캐시가 아니라 **로컬 Jekyll**을 봅니다. **`build.frontendDist`** 는 배포용으로 **GitHub Pages URL**을 유지합니다.

**`jekyll serve`와 “빌드”**: 예전에 말한 “매번 전체 빌드가 아니다”는 **`jekyll build`를 릴리스마다 통째로 돌리지 않는다**는 맥락에 가깝습니다. **`serve`는 파일을 저장할 때마다 재생성을 돌립니다.** 다만 **`--incremental`**(이 저장소의 `npm run dev:jekyll`·`tools/run.sh`에 포함)을 켜 두면 **바뀐 파일과 그에 의존하는 것만** 다시 만드는 쪽이라, 기본 `serve`보다 멈춤이 덜한 경우가 많습니다. 그래도 `_config.yml`·공용 레이아웃·대량 포스트를 건드리면 한동안 무거울 수 있습니다.

## 원격 + 캐시(오프라인에 가깝게)

- KarmoLab 페이지(`apps/karmolab/index.html`)는 **프로덕션 빌드에서** 사이트 루트의 **`/sw.min.js`(Chirpy 서비스 워커)** 를 등록합니다. 본문 레이아웃을 쓰지 않던 페이지라 기존에는 SW가 붙지 않았습니다.
- 그 SW는 설정상 **거부 경로가 아닌 GET 요청**을 네트워크로 받은 뒤 **Cache Storage에 넣습니다**. 그래서 **같은 출처**(`/karmolab/`, `/apps/karmolab/…` 등)는 방문·로드된 범위에서 캐시에 쌓일 수 있습니다.
- **한계**: (1) 최초 실행부터 오프라인이면 캐시가 없어 빈 화면/실패할 수 있습니다. (2) 브라우저·WebView2가 디스크를 비우면 캐시가 사라집니다. (3) **다른 도메인 CDN**은 CORS·opaque 응답 때문에 SW에 안 남거나 불완전할 수 있어, 일부 스크립트/폰트는 여전히 온라인이 필요할 수 있습니다.
- 로컬에서 앱으로 확인할 때는 위 **`npm run dev`** 흐름을 쓰면 됩니다. 배포본·서비스 워커·원격 캐시를 **개발 빌드(Rust 디버그)** 로만 검증하려면 **`npm run dev:remote`** (`src-tauri/tauri.dev-remote.conf.json`이 `devUrl`만 GitHub Pages로 덮어씀).

## 배포·원격 검증(짧은 체크리스트)

1. **`npm run dev:remote`** 로 WebView가 GitHub Pages `karmolab` 을 띄우는지.
2. **사이드바 → 기타 → 디버그** 에서 OS 알림 테스트(성공/에러 로그).
3. **트레이**: 창 숨김, 다시 실행 시 단일 인스턴스로 앞으로 오는지.
4. **서비스 워커**: `index.html` 의 SW 등록은 **프로덕션 Jekyll** 에만 들어갑니다. 로컬 `jekyll serve`(기본 development)로는 해당 스크립트가 빠지므로, SW·오프라인 캐시는 **배포 URL** 또는 `JEKYLL_ENV=production` 으로 빌드한 `_site` 로 확인하세요.

## 데스크톱 전용

- Tauri가 주입: `window.__KARMOLAB_DESKTOP__ === true` (웹 브라우저에는 없음)
- KarmoLab 스크립트: `Toolbox.isDesktopApp()` 로 동일 분기
- **단일 인스턴스**: 앱을 다시 실행하면 새 창 대신 기존 창이 앞으로 옵니다(최소화 해제 포함).
- **창 닫기(X)**: 앱이 종료되지 않고 트레이로 숨겨집니다. 완전 종료는 트레이 메뉴의 **종료**만 사용하세요.
- 트레이: **왼쪽 클릭**으로 메뉴(창 보이기 / 브라우저에서 열기 / 종료). Windows에서는 트레이 아이콘 **왼쪽 더블클릭**으로 메뉴 없이 창만 앞으로.
- 헤더에 **브라우저** 링크(앱 전용): 기본 브라우저로 같은 사이트를 엽니다.
- **OS 알림**: `showToast`가 에러·성공을 띄울 때(일부 짧은 성공 메시지 제외) 데스크톱 알림으로도 보냅니다. **에러 토스트**는 `sound: Mail`을 붙입니다(Windows에서 셸이 대소문자 정규화). 성공은 무음(기존과 동일)입니다.
- `devtools.js`는 **더 이상 Rust `include_str`로 앱에 박지 않습니다**(옛 스크립트가 최신 위젯·JSON UI를 덮어쓰던 문제 방지).
- **소리**: 토스트 자체는 Windows·집중 방해 설정에 따라 무음일 수 있어, `sound`가 켜진 요청은 토스트 표시 **직후** `MessageBeep`(시스템 별표음)으로 한 번 더 알립니다. **이 동작은 앱을 다시 빌드해야** 들어갑니다.
- **`desktop_notify` 인자** (`invoke('desktop_notify', { … })`): `title`, `body` 필수. 선택: `sound`, `image_path`(스네이크 케이스 — 로컬 파일 절대 경로).
  - **Windows** (`notify-rust` → WinRT 토스트): `sound`는 예) `IM`, `Mail`, `Reminder`, `SMS`, `Alarm`, `Call` … 무음이면 `silent` 또는 생략. 값 `Default`는 WebView 무음 이슈를 피하려고 셸에서 **`IM`** 알림음으로 바꿔 보냅니다.
  - **Linux/macOS**는 같은 필드가 있어도 데스크톱/세션에 따라 무시되거나 다르게 동작할 수 있습니다.
  - 배너·소리·방해 금지는 **Windows 설정 → 시스템 → 알림**에서 앱별로 조정합니다.

## 사이드바에 「디버그」가 없을 때

**브라우저에서는 보이는데 앱에서만 안 보일 때**는 WebView2가 **옛 캐시**를 쓰는 경우가 많습니다. 앱 데이터 삭제·강력 새로고침을 시도하거나, GitHub에 배포된 최신본과 비교해 보세요.
