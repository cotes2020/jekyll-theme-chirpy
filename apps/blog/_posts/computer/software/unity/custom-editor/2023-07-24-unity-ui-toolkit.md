---
title: "Unity - UI Toolkit"
# description: ""
categories: [컴퓨터, 소프트웨어]
tags: [유니티, UIToolkit]
image: "/assets/img/background/20240827-140647.jpg"

date: 2023-07-24. 22:30
# last_modified_at: 2024-03-27. 15:39
# last_modified_at: 2024-08-29. 21:50
last_modified_at: 2025-04-30. 07:00 # 메모
---

## UI Toolkit

---

기존 IMGUI, UGUI 와는 다른 또 다른 UI 시스템  
UI Elements가 유니티 2020 버전으로 넘어오면서 UI Toolkit으로 이름이 바뀜  

웹 개발의 `레이아웃`, `스타일`, `로직` 개념 그대로  
UI Toolkit에서는 레이아웃 - `UXML`, 스타일- `USS`, 로직 - `C#` 으로 UI를 구현  

런타임 UI, 에디터 UI 모두 구현 가능  

런타임 UI의 경우, UI Toolkit으로 구현된 UI는 마지막 단계에 렌더되기 때문에, 일단 월드 스페이스에는 UI를 보여줄 수 없는 것으로  

기존 에디터 UI들도 UI Toolkit으로 다시 만들어짐 (아마 2022버전부터?)  
그래서 UI Toolkit Debugger를 통해 기존 에디터의 UI를 직접 디버깅할 수도 있음  

- 필자는 커스텀 에디터 UI 구현을 목적으로 공부  
  - 하려고 했는데, 런타임 UI와 에디터 UI는 단지 보여지는 곳의 차이일 뿐, 구현은 똑같은 방식으로 하는 거였음  
  - 그래서 런타임 UI 용으로 만든 레이아웃, 스타일, 로직을 그대로 에디터 UI 에 적용시킬 수 있고, 반대도 마찬가지
  - [이때, 로직의 경우 2023부터 제대로 동작하는듯?](https://youtu.be/J2KNj3bw0Bw?t=2727)

## SerializedObject

---

Serialize된 데이터를 Unity에서 다루기 쉽게 가공한 것  

유니티 에디터 상의 모든 오브젝트는 SerializedObject로 변환되어 다루어짐  
UnityEngine Object/스크립트가 편집하는 영역 <-> SerializedObject/에디터가 편집하는 영역  

UnityEngine Object를 Asset으로 만들 때,  
UnityEngine Object는 Serialized Object로 변환된 이후, Serialized Object에서 Asset과 .meta파일 생성  

무튼 Editor에서도 SerializedObject를 다룸  

## SerializedProperty

---

`SerializedObject변수.FindProperty()`  

C#의 리플렉션을 통해,  
SerializedObject에서 SerializedProperty을 얻을 수 있음  

## VisualElement

---

UI Toolkit의 모든 Element들의 Base가 되는 Element  
VisualElement 자체는 아무 기능이 없고, 구체화된 VisualElement들의 단순 컨테이너 용으로 쓰임  

C#으로 치면 Object?  

모든 VisualElement는 generateVisualContext 콜백을 가짐  

## [Property Drawer](https://docs.unity3d.com/kr/2022.3/Manual/editor-PropertyDrawers.html)

---

컴포넌트/스크립트의 `속성`이 인스펙터에 보이는 방법을 제어/커스텀  

```cs
[CustomPropertyDrawer(typeof(Something))]
public class SomethingEditor: PropertyDrawer
{
    public override VisualElement CreatePropertyGUI(SerializedProperty property)
    {
        return new PropertyField(property);
        // 위 코드는 기존 모양 그대로 출력
    }
}
```

## [Custom Editor](https://docs.unity3d.com/kr/2022.3/Manual/editor-CustomEditors.html)

---

컴포넌트/스크립트가 인스펙터에 보이는 방법을 제어/커스텀  

```cs
[CustomEditor(typeof(Something))]
public class SomethingEditor: Editor
{
    public override VisualElement CreateInspectorGUI()
    {
        var root = new VisualElement();
        InspectorElement.FillDefaultInspector(root, serializedObject, this);
        // 위 코드는 기존 모양 그대로 출력
        return root;
    }
}
```

## viewDataKey

---

특정 VisualElement의 Unique한 값  
VisualElement의 상태를 저장하고 불러올 때 사용됨  

지정하지 않으면 VisualElement가 포함된 윈도우를 열거나 할 때마다 새로 생성 = 기본 상태  
지정하면 마지막 상태로 복구  

i.e.  

Foldout은 기본적으로 접혀진 상태  
만약 viewDataKey를 지정하면, 마지막으로 Foldout를 펼쳤을 경우 다시 윈도우를 열거나 할 때 접혀진 상태로 복구  

ScrollView의 경우, 마지막으로 스크롤한 위치를 복구한다던지 등  

## Foldout

---

접거나 펼칠 수 있는 박스  

```cs
var foldout = new Foldout()
{
    viewDataKey = "*Foldout",
    text = "인스펙터에서 보여질 Foldout Text",
    InspectorElement.FillDefaultInspector(root, serializedObject, this);
}
```

## UXML 연결하기

---

```cs
public VisualTreeAsset someUXML;

// CustomEditor라면
public override VisualElement CreateInspectorGUI()
{
    var root = new VisualElement();
    someUXML.CloneTree(root);

    // ...
}
```

Project 창에서 해당 Editor 스크립트를 선택하고, UXML 파일 할당  

## [커스텀 UI Shape?](https://youtu.be/J2KNj3bw0Bw?t=1367)

---

BindableElement  

[UxmlFactory](https://docs.unity3d.com/ScriptReference/UIElements.Image.UxmlFactory.html)
UXML 파일에서 불러온 데이터로 Image 인스터싱?  

generateVisualContext 콜백에 MeshGenerationContext를 받는 함수를 등록하면 메쉬 그릴 수 있음  
Unity에서 지원하는 Painter2D API 활용  

## Editor Window

---

```cs
public class SomethingEditor: EditorWindow
{
    [SerializeField] Something something;

    [MenuItem("SomePath/Something")]
    static void CreateMenu()
    {
        var window = GetWindow<SomethingWindow>();
        window.titleContent = new GUIContent("Complex");
    }

    public void OnEnable() { // ... }
    public void CreateGUI() { // ... }

    // https://youtu.be/J2KNj3bw0Bw?t=2519
}
```

## 메모

---

- ListView
  - makeItem
  - bindItem
- QuizU: Toolkit Sample

### 참고

- [Extending the Unity Editor with custom tools using UI Toolkit \| Unite 2022](https://www.youtube.com/watch?v=J2KNj3bw0Bw)
- [에디터 확장 입문 - 번역 5장 SerializedObject에 대해서](https://blog.naver.com/hammerimpact/220770624015)
- [참고](https://mechurak.github.io/2023-02-24_unity_ui_toolkit/)
- [참고](https://smilejsu.tistory.com/2317)
