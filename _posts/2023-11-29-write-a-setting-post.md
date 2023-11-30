---
title: Chirpy 설치/셋팅 방법
date: 2023-02-05 13:12:00 +0800
categories: [Blog, Blog설정/셋팅]
tags: [writing]
---
# Chirpy 테마를 사용한 Git 블로그 설정

저느 Git 기반 블로그를 빠르고 쉽게 설정할 수 있는 GitBlog을 사용하기로 결정했습니다. 시작하기 전에 아래의 세 가지 사이트에서 테마를 선택해야 합니다.
1. [Jekyll Themes](http://jekyllthemes.org/themes/monos/)
2. [Jekyll Themes Free](https://jekyll-themes.com/free)
3. [Jekyll Themes IO Free](https://jekyllthemes.io/free)

개인적으로 3번 사이트가 가장 깔끔하고 잘 정리되어 있다고 생각했습니다.

## 선택한 테마 : Chirpy

제가 선택한 테마는[Chirpy](https://chirpy.cotes.page/)을 선택했습니다. 제가 생각하기에 깔끔하고 여러 기능을 갖추고 있고, 쉽게 커스텀이 가능하다는 장점이 있고, 또한 많은 사람들이 사용하고 있다기에 선택 하였습니다. 저는 Fork 기준으로 사용하였기 때문에, Fork기준으로 말씀 드리겠습니다.

## 블로그 설정 단계 (Fork 기준)

1. **Repository Fork**
   - [Chirpy repository on GitHub](https://github.com/cotes2020/jekyll-theme-chirpy)로 이동합니다.
   - 오른쪽 상단의 "Fork" 버튼을 클릭하여 저장소를 본인 GitHub 계정으로 Fork합니다.
   - 반드시 [github ID].github.io 이 형식으로 Fork 하고 생성하셔야 합니다.

2. **Repository Clone**
   - 클론 명령을 사용하여 저장소를 로컬 머신으로 복제합니다.
     ```bash
     git clone https://github.com/githubname/jekyll-theme-chirpy.git
     ```

3. **Repository setting:**
   1. Fork한 Repository에서 Setting 창에 들어갑니다.
   2. Settings - General에서 master이름을 main으로 변경합니다.
   3. Settings - branch를 master에서 main으로 변경하고 Branch protection rule도 기본값(체크 X)으로 설정합니다.
   4. 배포 Settings - Pages - Build and deployment 에서 소스를 GitHub Actions로 변경합니다.
   5. Configure를 선택하고, 별도의 수정 없이 Commit changes…를 선택 후 Commit changes 선택합니다.
   6. .gihub - workflow 디렉토리 내에서 기존 배포 방식(Deploy form a branch)에 사용되던 파일을 삭제합니다.

참고 :: GitHub Actions로 소스를 변경하지 않거나, Configure를 완료하지 않고 배포할 경우 index.html 화면만 표시되니 주의합니다.


4. **Local Repository:**
   7. Github에서 jekyll.yml을 생성했으므로 git pull을 통해 로컬 리소스와 동기화를 먼저 진행합니다.
   8. .gitignore 내 assets/js/dist 디렉토리 내 파일들의 Push가 무시되도록하는 설정을 주석처리 합니다.
   9. git 배포를 위해 _posts 경로에 테스트용 포스트를 생성한 후 git push 합니다.

5. **Repository Check:**
   10. Github - Actions 탭에서 배포 워크플로우 실행을 확인할 수 있습니다.
   11. 테스트 페이지 및 블로그 기능이 정상 동작하는지 확인합니다.

