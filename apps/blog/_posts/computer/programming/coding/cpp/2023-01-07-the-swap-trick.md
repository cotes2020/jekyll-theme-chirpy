---
title: "The Swap Trick"
# description: ""
categories: [컴퓨터, 프로그래밍]
tags: []
image: "/assets/img/background/kururu-lab.jpg"

date: 2023-01-07. 23:22
# last_modified_at: 2023-01-07. 23:22
---

## The Swap Trick: 메모리 재할당

---

[참고0](https://d-yong.tistory.com/74)  
[참고1](https://www.appsloveworld.com/cplus/100/357/the-swap-trick-stl)  
[참고2](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=sorkelf&logNo=221039099285)
[참고3](https://sorting.tistory.com/9)

읽기 좋은 코드가 좋은 코드다의 84p 에서 언급된 The Swap Trick.  
아래는 책에 나온 예제 코드다.  

```cpp
struct Recorder
{
    vector<float> data;
    ...
    void Clear()
    {
        vector<float>().swap(data);
        // 뭐? 그냥 data.clear() 를 호출하지 않는 이유가 뭐지?
    }
}
```

위 코드를 보면 주석에 적힌 말처럼 단순히 data.clear() 호출하면 될 것 같은데,  
빈 벡터 (vector<float>()) 와 swap 해주고 있다.  

이는 '잘 알려지지 않은 C++ 언어 특유의 세부 사항 ㅡ 저자에 따르면' 을 알아야 한다.  

vector.clear()는 vector 안에 저장된 값들은 제거시켜주지만,  
vector에 할당된 메모리는 해제되지 않는다!  

때문에 강제로 메모리를 해제해주기 위해,  
빈 vector와 swap을 해주는 것이다.  

스코프가 끝나는 시점에는 자동으로 힙에서 메모리가 해제되기 때문에,  
계속해서 사용하는 vector가 아니라면 이렇게 쓸 필요는 없을 것 같다.  

책에서 나오는 The Swap Trick 은 이를 의미하는 것 같고,  
검색해보니 다른 것도 있는 것 같다.  

```cpp
vector<float>(data).swap(data);
```

이렇게 같은 vector를 그대로 swap 해주면,  
vector가 메모리를 딱 저장된 요소들만큼만 사용하게 된다.  

무슨 말이냐 하면,  

vector는 용량이 꽉 찼을때 스스로 메모리를 재할당하여 일정 비율로 크기를 키우는데,  
이때 커진 메모리를 모두 사용하지 않는 이상, 낭비되는 메모리가 생기게 된다.  

때문에 위와 같은 방법으로 The Swap Trick을 사용해 메모리를 절약한다.  
이는 vector 뿐만 아니라 string, deque 같이 동적으로 메모리 할당량이 바뀌는 친구들에게도 적용된다고 한다.  

C++ 11 에서는 이와 같은 기능을 하는 shrink_to_fit() 함수가 있다고 하는데,  
이도 새로운 vector를 만들고, 복사하는 것이기에,  
큰 vector에 대하여 사용할 경우, CPU 오버헤드를 한 번 고민해봐야 한다.  

또, 방금 언급한 것 처럼 과도한 오버헤드가 발생할 수 있기에,  
shrink_to_fit() 함수는 non-binding 함수 (모든 컴파일러에서 반드시 구현되지는 않는) 이라고 한다.  
때문에 이전에 사용하던 컴파일러와의 호환성 역시 생각해야 할 것이다.  
