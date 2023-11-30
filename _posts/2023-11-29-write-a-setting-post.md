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

![Desktop View](/LJW22222.github.io/assets/img/favicons/posts/githubforkimageone.png){: width="972" height="589" }

3. **Install Dependencies:**
   - Navigate to the cloned repository and install dependencies:
     ```bash
     cd jekyll-theme-chirpy
     bundle install
     ```

4. **Configure Site Information:**
   - Open the `_config.yml` file and update site information.

5. **Create and Edit Posts:**
   - Use the `/_posts` directory to create or modify blog posts in Markdown.

6. **Preview Your Site:**
   - Build and serve your site locally:
     ```bash
     bundle exec jekyll serve
     ```
   - Open [http://localhost:4000](http://localhost:4000) in your browser to preview the blog.

7. **Customize the Theme:**
   - Customize the theme by modifying HTML, CSS, or other assets.

8. **Publish to GitHub Pages:**
   - Deploy your blog to GitHub Pages or any other hosting service.

9. **Update and Maintain:**
   - Regularly update and maintain your blog with new content and software updates.

Good luck with your tech blog journey!
