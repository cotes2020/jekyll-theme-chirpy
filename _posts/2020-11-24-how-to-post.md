---
title: 글을 업로드하는 방법
author: Shin Hyun
date: 2020-11-24 21:40:00 +0900
categories: [Barami,Others]
tags: [post,shinhyun]     # TAG names should always be lowercase, 띄어쓰기도 금지
---

# 글을 올리는 방법

1. <a href="https://github.com/hyu-barami" target="_blank">hyu-barami</a> 가입 신청을 한다. 

    가입 신청은 신 현, 혹은 박제윤에게 본인 깃헙 아이디를 알려주면 추가하겠다. 메일이 가면 해당 메일로 온 링크 클릭하면 된다. 

2. 깃헙을 클론한다.
```
git clone https://github.com/hyu-barami/hyu-barami.github.io.git
```

3. 본인이름으로 브랜치를 만든다.
```
git checkout -b (본인이름영어로)
# Example
git checkout -b shinhyun
```

4. _posts/ 에 글을 쓴다.

5. 작성한 글을 커밋하고
```
git add ./_posts/(해당파일명) 
git commit -m "Add: Post (파일명 or title명)"
git push --set-upstream origin (해당브랜치명)
# Example
git add ./_posts/2020-11-24-2020-exhibition-example.md
git commit -m "Add: Post 글을 업로드하는 방법"
git push --set-upstream origin shinhyun
```

6. 풀리퀘스트를 보낸다.

    1. 아래 버튼을 클릭하자. 
    
        <img src="/assets/img/post/2020-11-24-how-to-post/pull-request.PNG" width="90%"> 
        
        안 보이면, branches 들어가서, 해당 브랜치의 New Pull Request 버튼을 클릭하자. 
    
    2. 풀리퀘 보내는 대상 레포지토리를 hyu-barami로 변경해준다. 
    
        실수로 지킬 템플릿 제작자에게 풀리퀘를 보내지 않도록 주의하자.
        
        <img src="/assets/img/post/2020-11-24-how-to-post/pull-request2.PNG" width="90%"> 
       
    3. 풀리퀘스트를 보낸다. 
    
        제목과 내용은 별다른 양식은 없지만, 최소한 어느 포스트를 올렸는지에 대해 알아서 간략하게 작성해주자. 
        <img src="/assets/img/post/2020-11-24-how-to-post/pull-request3.PNG" width="90%"> 
        
    4. Able to Merge 뜨면 머지해도 되고, 아니라면 신 현에게 도움을 요청하자 ! 
    
        Able to Merge 뜨더라도 한 번 확인받고 보내고 싶다면 신 현에게 도움을 요청하자 ! 
        
        Post 업로드/수정 관련이 아닌, 다른 수정사항이라면 머지하지 말고 신 현에게 노티를 주도록 하자. 