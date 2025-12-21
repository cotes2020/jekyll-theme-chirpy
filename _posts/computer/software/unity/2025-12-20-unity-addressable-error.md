---
title: "Unity Addressable Asset 이름 설정 문제"
# description: ""
categories: [컴퓨터, 소프트웨어]
tags: [유니티]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2025-12-20. 15:31 # Init
# last_modified_at: 2025-12-20. 15:31
---

## 문제

---

![251220-152830](/assets/img/post/computer/251220-152830.png)

```cs
Exception thrown in DynamicInvoke: System.Reflection.TargetInvocationException: Exception has been thrown by the target of an invocation. ---> System.NullReferenceException: Object reference not set to an instance of an object
  at UnityEngine.ResourceManagement.ResourceProviders.AssetDatabaseProvider.LoadAssetSubObject (System.String assetPath, System.String subObjectName, System.Type type) [0x0000f] in .\Library\PackageCache\com.unity.addressables@45e9abf44299\Runtime\ResourceManager\ResourceProviders\AssetDatabaseProvider.cs:24 
  at UnityEngine.ResourceManagement.ResourceProviders.AssetDatabaseProvider+InternalOp.LoadImmediate () [0x000b1] in .\Library\PackageCache\com.unity.addressables@45e9abf44299\Runtime\ResourceManager\ResourceProviders\AssetDatabaseProvider.cs:108 
  at (wrapper managed-to-native) System.Reflection.RuntimeMethodInfo.InternalInvoke(System.Reflection.RuntimeMethodInfo,object,object[],System.Exception&)
  at System.Reflection.RuntimeMethodInfo.Invoke (System.Object obj, System.Reflection.BindingFlags invokeAttr, System.Reflection.Binder binder, System.Object[] parameters, System.Globalization.CultureInfo culture) [0x0006a] in <1eb9db207454431c84a47bcd81e79c37>:0 
   --- End of inner exception stack trace ---
  at System.Reflection.RuntimeMethodInfo.Invoke (System.Object obj, System.Reflection.BindingFlags invokeAttr, System.Reflection.Binder binder, System.Object[] parameters, System.Globalization.CultureInfo culture) [0x00083] in <1eb9db207454431c84a47bcd81e79c37>:0 
  at System.Reflection.MethodBase.Invoke (System.Object obj, System.Object[] parameters) [0x00000] in <1eb9db207454431c84a47bcd81e79c37>:0 
  at System.Delegate.DynamicInvokeImpl (System.Object[] args) [0x000e7] in <1eb9db207454431c84a47bcd81e79c37>:0 
  at System.MulticastDelegate.DynamicInvokeImpl (System.Object[] args) [0x00008] in <1eb9db207454431c84a47bcd81e79c37>:0 
  at System.Delegate.DynamicInvoke (System.Object[] args) [0x00000] in <1eb9db207454431c84a47bcd81e79c37>:0 
  at UnityEngine.ResourceManagement.Util.DelayedActionManager+DelegateInfo.Invoke () [0x00000] in .\Library\PackageCache\com.unity.addressables@45e9abf44299\Runtime\ResourceManager\Util\DelayedActionManager.cs:46  184 (target=UnityEngine.ResourceManagement.ResourceProviders.AssetDatabaseProvider+InternalOp) InternalOp.LoadImmediate() @19.45547
UnityEngine.ResourceManagement.Util.DelayedActionManager:LateUpdate () (at ./Library/PackageCache/com.unity.addressables@45e9abf44299/Runtime/ResourceManager/Util/DelayedActionManager.cs:162)
```

오랜만에 프로젝트 열고 Play 눌렀는데, Addressable 불러오는 과정에서 Error가 발생했다. Addressable 불러오는 과정을 로딩 화면으로 만들었었는데, 로딩 화면 한 93%에서 게임이 멈춰서 진행이 불가능해졌다.  

## 원인

---

Addressable에 포함된 Asset 중 이름이 `SKL_5_슬라임탄환[TEMP]` 라는게 있었는데, Addressable에서 `[`나 `]` 같은 기호 들어가 있으면 못 불러오나보다.  

이름 수정하니까 Error 없이 잘 불러와진다.  

## 해결 과정

---

우선 코드 보기전에, 오랜만에 프로젝트를 열기도 했고, Unity 버전도 마구잡이로 올렸어가지고, Addressable 빌드 자체의 문제일 수 있을 것 같았다. 그래서 빌드 Clear하고 다시 돌려봤다.  

하지만 여전히 Error가 발생했다.  

StackTrace도 Addressable 불러올 때 문제 생긴다는 것 말고는 못알아먹겠고, Error 로그 복사해서 구글링 해봐도 뭐가 안나와서 코드를 보기 시작했다.  

로그를 보면 일단 가장 먼저 Reflection 쪽에서 로그를 남긴 걸 볼 수 있는데, 마침 내 프로젝트에서 Addressable 불러올 때 Reflection 쓰는 코드가 딱 한 군데 있어서, 그쪽에서 Error가 났음을 바로 알 수 있었다.  

```cs
private void LoadAssetsAsync(List<AsyncOperationHandle> handles)
{
    foreach (Type type in DataSODefine.AssetPrefixes.Keys)
    {
        LoadAssetByType(type, handles);
    }

    void LoadAssetByType(Type type, List<AsyncOperationHandle> handles)
    {
        MethodInfo method = typeof(DataLoader).GetMethod(nameof(LoadAsset), BindingFlags.Instance | BindingFlags.NonPublic);
        MethodInfo genericMethod = method.MakeGenericMethod(type);
        genericMethod.Invoke(this, new object[] { type.Name, handles });
    }
}

private void LoadAsset<T>(string label, List<AsyncOperationHandle> handles) where T : DataSO
{
    var handle = Addressables.LoadAssetsAsync<T>(label, null);
    handle.Completed += OnAssetsLoaded;
    handles.Add(handle);
}
```

위 코드다. 우선 이 코드의 가장 큰 목적은 각 Label 별로 Asset을 Addressable로 불러오는 것이다.  

곁가지 목적으로는 각 Label 로드마다 handle을 가져와서, 로딩 화면에서 로딩 진행도 (ex. 93%)를 보여주는 것이다. (handles에 넣어주면, `LoadAssetsAsync`를 호출한 곳에서 사용한다.)  

Addressable로 불러올 Label 목록은 `DataSODefine.AssetPrefixes.Keys` 이라는 곳에서 가져오고 있다. 이 프로젝트에선 Label 이름을 각 ScriptableObject의 Class 이름과 동일하게 지어놓은 특수한 상황이다. 때문에 이 맥락에선 `Type`이 곧 Label이다. `DataSODefine.AssetPrefixes.Keys` (`List<Type>`)는 불러올 `Type` (Label) 목록이다.  

`DataSO`는 모든 ScriptableObject가 상속받는 base Class다. `string Name`, `int ID` 같이 common하게 쓰이는 필드들이 들어있다.  

`LoadAssetByType` 로컬 메서드는 각 `Type` 별 Asset으로 추가적인 작업을 하는 `LoadAsset`을 호출한다. Reflection으로 각 `Type` 별 제네릭 메서드를 만들어 호출하는데, 이렇게 하지 않으면 아래처럼 각 `Type` 마다 호출을 하드코딩 해야해야 하기 때문에... Reflection을 썼다.  

```cs
LoadAsset<QuestSO>(nameof(QuestSO), handles);
LoadAsset<CardData>(nameof(CardData), handles);
LoadAsset<ItemData>(nameof(ItemData), handles);
LoadAsset<SkillData>(nameof(SkillData), handles);
...
```

`LoadAsset`에서는 각 Asset을 사용하기 좋게 `Dictionary<Type, DataSO>` 같은 구조로 저장한다. 이때 Type 정보가 필요해서 제네릭으로 만들었는데, 제네릭 메서드 호출 할 때 넘겨주는 `Type`을 `LoadAsset<type>(~)` 같이 변수로 넘겨줄 수가 없어서... Reflection을 썼다.  

아무튼 코드 설명은 그렇다.  

여기서 일단 각 코드 블럭마다 `Debug.Log`를 추가해서, 대충 어느 시점에 문제가 생기는지 간단히 확인했다. 각 Label 별로 호출되는 메서드가 있기 때문에, 여기에 로그를 추가하면 어느 Label에서 문제가 생기는지 알 수 있다.  

확인 결과 `SkillData` 로드 시에 Error가 발생했다.  

이번엔 어떤 Asset에서 문제가 생기는지 확인했다. 같은 `Type`의 Asset을 한 번에 불러오는 대신, 각 Asset을 하나하나 불러오는 방식으로 바꾼다.  

```cs
Addressables.LoadResourceLocationsAsync(label).Completed += (obj) =>

foreach (var location in obj.Result)
    var handle = Addressables.LoadAssetAsync<T>(location);
```

확인 결과 `SKL_5_슬라임탄환[TEMP]` ScriptableObject를 불러올 때 문제가 생겼다. Inspector에서 확인해보니 다른 `SkillData`랑 비교해봐도 특별히 다른 점은 없고, 굳이 꼽자면 이름에 `[TEMP]`가 있다는 것이었다.  

파일 이름이 문제나 되나 싶어 이름에서 `[TEMP]`을 지우니.. Error가 사라졌다.  

Addressable은 기호 들어간 파일을 불러오지 못하는 것일까?  

확인해보니 Addressable이 로컬 컴퓨터나 컴퓨터로 Web 요청보내다 보니까, 보통 URL 쓸 때처럼 알파벳이랑 `-`, `_` 말고는 쓰지 않는게 좋다고 한다.  

### 추가 조사

다른 방법이 있나 더 찾아보니, 이름말고 GUID로 불러오는 방법도 있다고 한다. ([참고](https://discussions.unity.com/t/unable-to-load-asset-when-asset-filename-is-accentuated-addressables/1578597/3))  

![251220-163914](/assets/img/post/computer/251220-163914.png)

기본적으로 Addressable Asset Settings에서 Internal Asset Naming Mode가 Full Path로 되어있는데, 이걸 GUID로 바꿀 수 있다.  

그런데 나는 내 환경에서는 여전히 Error가 발생했다. 구체적으로는 Addressable Group 창에서 설정으로 Addressable을 Addressable 빌드로 불러오게 하면 문제 없는데, 프로젝트 파일 직접 불러오게 하면 문제가 생긴다.  

그래서 그냥 파일 이름 바꾸는 방법으로 해결했다.
