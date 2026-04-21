---
title: "Unity HideFlags"
# description: ""
categories: [컴퓨터, 소프트웨어]
tags: [유니티]
image: "/assets/img/background/20240827-140647.jpg"

date: 2025-04-16. 19:20 # Init
# last_modified_at: 2025-04-16. 19:20
---

## 머리말

---

HideFlags  

## HideFlags

---

Hierarchy 창에서 보이지 않는 GameObject, Project 창에서 보이지 않는 Asset을 만들 수 있다.  
보이지는 않지만 실제로 존재하는 GameObject, Asset.  

### GameObject

`HideFlags.HideInHierarchy`  
`gameObject.hideFlags = HideFlags.HideInHierarchy;`  

Scene Asset을 Text Editor로 열어보거나, `GameObject.Find()` 등을 써보면 숨겨진 GameObject를 확인할 수 있다.  

Hierarchy 창에서 보이지 않지만, Scene 창에서는 여전히 보인다.  
Scene창에서 Click을 통해 선택되지는 않는다. (Selection)  
Code를 통해서는 선택 가능하다. `Selection.activeGameObject = gameObject`  

### Asset

`HideFlags.HideInHierarchy`  

메인 에셋은 에셋으로 인식되지 않아 의존 관계에 불편이 생길 수 있다. (?)  
서브 에셋은 가능하다.  

```cs
// [MenuItem ("Assets/Create SubAssets")]
AssetDatabase.CreateAsset (first, path);
// 서브 에셋 만들기
AssetDatabase.AddObjectToAsset (second, first);
AssetDatabase.ImportAsset (path);
```

## 메모

---

### 참고

- ['해머임팩트': '[에디터 확장 입문] 번역 27장 HideFlags Viewer'](https://blog.naver.com/hammerimpact/220780954105)
- ['Unity Documentation': 'HideFlags'](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/HideFlags.html)
