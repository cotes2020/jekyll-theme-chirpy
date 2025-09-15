---
title: DAELoader클래스의 개선작업 진행
description: 지금의 DAELoader라는 클래스는 사실 객체화가 정말 안되어있습니다..
author: ounols
date: '2021-05-23 20:00:00 +0800'
categories:
- Dev
- 자체 게임 엔진 프로젝트
tags:
- Coding
- dev
pin: false
math: true
mermaid: true
---

지금의 `DAELoader`라는 클래스는 사실 객체화가 정말 안되어있습니다..ㅎㅎ;;

예전에 DAE파일 파서 만들면서 구현 중심으로 제작하다보니 발생한 일이였지만<br/>
다른 요소들을 구현한다고 더 신경을 쓰지 못한 문제도 있었네요ㅠㅜ

<br/>

그러다 이번에 한 파일에 여러개의 매쉬를 가지고 있는 경우에 대해<br/>
정상적인 파싱작업이 이루어지도록 작업을 진행할 예정입니다.

```cpp
 private:
        const XNode* m_root{};
        MeshSurface* m_obj;

        std::vector<Vertex*> m_vertices;
        std::vector<vec3> m_normals;
        std::vector<vec2> m_texUVs;
        std::vector<int> m_indices;

        std::vector<float> m_f_vertices;
        std::vector<float> m_f_normals;
        std::vector<float> m_f_texUVs;
        std::vector<int> m_f_jointIDs;
        std::vector<float> m_f_weights;

        SkinningData* m_skinningData = nullptr;
        Skeleton* m_skeletonData = nullptr;
        DAEAnimationLoader* m_animationLoader = nullptr;
```
지금은 클래스 내부를 보면 DAE 로더를 통해 생성되는 요소의 객체화가 전혀 되지 않았습니다..ㅎㅎ;;<br/>
이번에 좀 한 객체로 묶어서 좀 더 관리하기 편하도록 만들어야겠습니다.

여러개의 애니메이션에 대한 요소도 존재하지 않아 이 부분도 어느정도 고민하면서 제작할 것 같네요ㅎㅎ


> 📣 관련 프로젝트 Git 주소 : [https://github.com/ounols/CSEngine](https://github.com/ounols/CSEngine){:target="_blank"} 
{: .tip}
