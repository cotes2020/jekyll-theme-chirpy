---
title: input의 onchange 그리고 match()
author: juyoung
date: 2020-10-16 18:56:00 +0800
categories: [project, process]
tags: [project]
---

# 검색창 기능 구현 순서

1. input enter만으로 value값 받기
2. input값과 json data 비교하기
3. input값이 해당 json data에 있을 때 그 목록으로 ulEle에 값을 넣어준다 
3. input값이 바뀌면 match로 얻은 값 새로고침하기
  

# 구현순서:
1. <form> 테그를 적용하고 addEventListener로 submit 콜백함수를 불러온다.   

2. 이때 e.preventDefault()로 input값을 넣고 enter할 때마다 페이지가 새로고침되지 않도록 막아야 input의 value를 얻을 수 있다.  

3. match() method를 사용하여 json의 영어, 또는 한국어 이름과 input의 value를 비교한다.
  
   대소문자 구분 없이 비교하려면 정규표현식으로 /비교하려는 값(input value)/gi 작성해야한다고 하는데 도저히 안되서 포기했다.  

5. match했을 때 같은 단어가 들어간 이름의 목록이 object형식으로 주어진다.

```console
 ["탱고", index: 0, input: "탱고 오나다", groups: undefined]  

0: "탱고" 
groups: undefined
index: 0
input: "탱고 오나다"
length: 1
__proto__: Array(0)
```

 6.inputVal에 바로 input.value를 할당하면 페이지가 새로고침될 때까지 input에 검색어를 바꿔주더라도 계속 ulEle에 목록이 쌓여 표현되는 문제가 발생했다.  

 이를 위해 input에 onChange()함수가 필요하다. 
<br> inputVal = e.target.value와 같이 target의 value를 할당하면 input에 검색어가 달라지면 enter를 쳤을 때 change 콜백함수가 실행되며 전 value값이 삭제되는 것 같다. 이로써 ulEle에 쌓이는 목록이 검색어가 달라질 때마다 매번 새로워진다.  

 7.input창에 입력을 마치고 enter를 치면 input창에 커서가 focus되고 빈칸으로 바뀌도록 할 때도 inputVal=''; 이라고 하면 검색한 값이 계속 남아있는데 반해 input.value='';라고 선언하면 빈칸이 된다.



```html
    <form>
            <div class="inputarea f_b">
                <input type="text" placeholder="밀롱가 이름으로 검색해보세요" 
                >
                <button style="display:none"></button>
                <div>
                    <a href="#">
                        <img src="img/ic_search.png" alt="">
                    </a>
                </div>
            </div>
        </form>
```

```js
 form.addEventListener('submit',dataFun);
   
   
     function dataFun(e) {
    e.preventDefault();
        response = JSON.parse(data.responseText);
        
        ulEle.innerHTML = '';
        input.addEventListener('change',function(e){
            inputVal = e.target.value;
           
        });
       
       
     
        response.millonga.forEach(function (el, idx) {
            thumb = el.thumb;
            url = el.url;
            en = el.en;
            address = el.address;
            ko = el.ko;
           
          
            let a = en.match(inputVal);
            let b = ko.match(inputVal);
             console.log(a || b);
                               
                  if (a || b) {
                 
                    liEle = "<li class='item item" + idx + " f_b'>";
                    liEle += "<div class='con f_b'> <div class='leftsec'><div class='thumb'><a class='linkA link" + idx + "' href='" + url + "'><img src='" + thumb + "' alt='" + en + "'></a></div></div>";
                    liEle += " <div class='rightsec'> <div class='f_b'><h4 class='f_b'>" + en + "</h4><span>거리m</span></div><h6>" + ko + "</h6>";
                    liEle += " <p class='address'>" + address + "</p></div> </div>";
                    liEle += " <div class='appraisal'><span class='like'>371</span><span class='write'>39</span> </div></li>";
                    ulEle.innerHTML += liEle;
                   
                    input.value = '';
                    input.focus();  
                  }else if(!a || !b){
                    input.value = '';
                    input.focus();  
                  }
               
                 
                
                 
            });
           
               
       

    }//datafun

```



[^footnote]: 다음에 해결해야 할 문제
이렇게 목록을 받아오면 drag에서 mouseup event가 중복 발생하는 bubbling 문제가 발생했다.  

