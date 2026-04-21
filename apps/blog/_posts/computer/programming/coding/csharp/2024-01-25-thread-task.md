---
title: "C# Thread/Task"
# description: ""
categories: [컴퓨터, 프로그래밍]
tags: [CSharp]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2024-01-25. 05:42
# last_modified_at: 2024-04-27. 09:24
# last_modified_at: 2024-08-29. 21:54
last_modified_at: 2024-08-30. 00:06
---

## [lock](https://learn.microsoft.com/ko-kr/dotnet/csharp/language-reference/statements/lock)

---

```cs
private readonly object thisLock = new object();
lock (thisLock)
{
    // Bla Bla
}
```

임계 영역 (`Critical Section`)을 만들어주는 키워드  
`Thread`는 `lock`을 얻어야 `Critical Secion`을 생성할 수 있다.  

외부 코드에서도 접근할 수 있는 `this`, `Type` 형식 (`typeof`, `GetType()`), `string` 형식은 매개변수로 절대 사용하지 말 것.  

@ Key, 전용 개체 인스턴스  
@ `Critical Section`, 한 번에 한 스레드만 접근할 수 있는 코드 영역  

## 스레드

---

[스레드](/posts/process-processor-thread/)

### 스레드는 가벼운 자원인가?

상대적으로 프로세스보다 가볍지만,  
절대적으로 무거운 리소스이다.  

- 공간비용
  - `Thread Kernel Object`
    - x86: 700B
    - x64: 1240B
    - ARM: 350B
  - `Thread Environment Block`: 4K
  - `User Mode stack`: 1MB
  - `Kernel Mode stack`
    - 32bit OS: 12KB
    - 64bit OS: 24KB
  - 총: 1053KB 남짓

- 시간비용
  - DLL Thread attach/detach notification
    - 프로세스에 스레드를 만들어질때마다, 각 DLL의 main 함수를 호출
    - 문안인사를 하는 것이지요, 스레드 새로 만들어졌어요
    - DLL은 스레드를 위한 공간을 마련
    - 대표적인게 C Runtime Library (DLL)  
  - `Context Switching`
    - 이 스레드가 갖고 있던 가상의 CPU 정보들을 로드해서 수행하고
    - 일정 시간 (Quantum)이 지나면 다시 저장하고
    - 다음 스레드를 로드...
  - -> DLL이 많으면 많을수록, 스레드가 많으면 많을수록

### 이제 그만 해야 할 바보짓

![작업관리자](/assets/img/post/stone/2024/240427_00.png)  

### 명시적으로 스레드를 생성하지 말라

- 예외
  - `보통` 단계의 스레드 우선순위가 아닌 스레드가 필요한 경우
  - 포그라운드 스레드처럼 동작하는 스레드가 필요한 경우
  - 계산 중심의 작업이 상당히 오랫동안 수행되어야 하는 경우

가능한 Thread class를 이용하여 명시적으로 스레드를 생성하지 말 것  

### 여러 스레드를 사용하는 이유

- 응답성의 개선
  - 클라이언트 측 UI 어플리케이션
  - UI 스레드
  - 작업 스레드의 개수는 늘어나지만, 응답성이 개선되므로 전체적으로 좀 더 나은 응용 프로그램으로 판단

- 성능
  - 클라이언트, 서버 측 Application
  - 다중 CPUI에 한해서 성능 개선

- 스레드를 가장 잘 화용하는 방법
  - 스레드 풀을 이용하고 비동기로 작업을 수행하라
  - -> 우리가 TASK 병렬화를 알아야 하는 이유

### 계산 중심 비동기 작업

- CLR 스레드 풀
  - 사실 이미 만들어져 있음
  - 명시적으로 쓰지않고 있었을 뿐
  - Requests Queue에 작업이 들어오면, 그때 스레드를 할당
  - 최대한 하나만 쓰려고함
    - 많이 들어오면, 코어 수만큼만 스레드를 만들어서 사용
  - Request Queue에 작업이 일정시간 들어오지 않으면 스레드를 제거
    - 생성/제거 비용 최소화

### QueueUserWorkItem

```cs
public static bool QueueUserWorkItem(WaitCallback callBack, object state);
```

```cs
for (int i = 0; i < 100; i++)
{
    ThreadPool.QueueUserWorkItem((obj) =>
    {
        Console.WriteLine(Thread.CurrentThread.ManagedThreadId);
    });
}

Console.ReadLine();
```

- 작업 완료 시점을 알 수 없음
- 작업 수행 결과를 얻어 올 수 없음
- 취소 / 예외 처리 불가능
- -> 쓰기 쉽지만 이런 한계들 때문에 잘 안씀

### Task

Task 나누면 장접  
Taks 간의 상관관계가 없다면, 병렬로 수행 가능 -> 빨라진다  

```cs
// QueueUserWorkItem과 유사 동작을 수행하는 코드 패턴
Action action = () =>
{
    Console.WriteLine(Thread.CurrentThread.ManagedThreadId);
};

Task task = new Task(action); // #1: Task 객체 생성 후
task.Start(); // Start 명시적 호출

Task.Run(action); // #2: Task.Run을 이용하여 작업 수행
```

```cs
// 결과 값을 가져오는 Task 객체 생성, Sum 호출 시 예외가 발생한다면?
Task<int> task = new Task<int>((n) => Sum((int)n), 100);
t.Start(); // 명시적 수행
t.Wait(); // Task 완료 대기 (완료 시점을 알 수 있다)

Console.WrtieLine(t.Result); // t.Result 결과 획득 (결과를 받아올 수 있다)
```

Canceling a Task  

```cs
private static int Sum(CancellationToken ct, int n)
{
    int sum = 0;
    for (; n > 0; n--)
    {
        // 작업 취소가 요청되면 OperationCanceledException을
        // innerException으로 갖는 AggregateException을 던짐
        ct.ThrowIfCancellationRequested();
        checked
        {
            sum += n;
        }
    }
    return sum;
}

static void Main(string[] args)
{
    CancellationTokenSource cts = new CancellationTokenSource();
    Task<Int32> t = Task.Run(() => Sum(cts.Token, 100000000), cts.Token);
    cts.Cancel(); // 작업 취소 요청

    try
    {
        Console.WriteLine("The result is: " + t.Result);
    }
    catch (AggregateException e)
    {
        e.Handle((innerException) => innerException is OperationCanceledException);
        // Operation.. 이면 처리된 것으로  

        Console.WriteLine("Exception: " + e.InnerExceptions[0].Message);
    }
}
```

- 작업 완료 시점을 알 수 있음
- 작업 수행 결과를 얻어 올 수 있음
- 취소 / 예외 처리 가능

### Task 연결1

```cs
// 웨이팅 아키텍처
Task<int> task = new Task<int>((n) => Sum((int)n), 100);
t.Start();
t.Wait(); // 대기

Console.WrtieLine(t.Result);
```

to  

```cs
// waitfree, lockfree 아키텍처 (서버에서 많이 사용)

// t Task가 완료되면 cwt Task를 수행
Task<Int32> t = Task.Run(() => Sum(CancellationTokenSource.None, 100));
Task cwt = t.ContinueWith( // 완료되면
(antecedent) =>
{
    Console.WriteLine("The result is: " + antecedent.Result);
});

// 연결하고 바로 빠져나옴
```

### Task 연결2

```cs
// TaskContinuationOptions
// OnlyOnCanceled, OnlyOnFaulted, OnlyOnRanToCompletion 그외 기타 등등

CancellationTokeSource cts = new CancellationTokenSource();
cts.Cancel();

Task<Int32> t = Task.Run(() => Sum(cts.Token, 100000000), cts.Token);

t.ContinueWith((task) => // 성공 완료시
{
    Console.WriteLine("The result is: " + task.Result);
}, TaskContinuationOptions.OnlyOnRanToCompletion);
t.ContinueWith((task) => // 실패/예외 발생 시
{
    Console.WriteLine("The task failed" + task.Exception.InnerException);
}, TaskContinuationOptions.OnlyOnFaulted);
t.ContinueWith((task) => // 취소시
{
    Console.WriteLine("The task was canceled");
}, TaskContinuationOptions.OnlyOnCanceled);
```

### Task 연결3

```cs
// Parent-Child Task로의 연결, TaskCreationOptions.AttachedToParent
Task<Int32[]> parent = new Task<Int32[]>(() =>
{
    var results = new Int32[3];
    new Task(() => results[0] = Sum(CancellationToken.None, 10000), TaskCreationOptions.AttachedToParent).Start();
    new Task(() => results[1] = Sum(CancellationToken.None, 20000), TaskCreationOptions.AttachedToParent).Start();
    new Task(() => results[2] = Sum(CancellationToken.None, 30000), TaskCreationOptions.AttachedToParent).Start();
    return results;
});

// Child Task들이 모두 완료되면 = parent가 완료되면

var cwt = parent.ContinueWith((parentTask) => // parentTask가 끝나면 수행할 Task 연결
{
    Array.ForEach(parentTask.Result, Console.WriteLine);
});

parent.Start();
```

### I/O 중심의 비동기 작업

### 동기 I/O 매커니즘

### 비동기 I/O 매커니즘

- 비동기
  - 작업을 하는 주체와 작업을 요청하는 주체가 다름
  - 작업을 하는 주체가 작업을 요청하는 주체에게 알려주는 방식
    - (H/W가 완료하면 스레드에게 노티피케이션을 줌)

이를 어떻게 패턴화하느냐?  

### Comparing Patterns

- Sync

```cs
public int Read(byte[] buffer, int offset, int count);
// 근데 비동기가 좋잖아?
```

- APM (Asynchronous Programming Model)

```cs
public IAsyncResult BeginRead(byte[] buffer, int offset, int count, AsyncCallback callback, object state);
public int EndRead(IAsyncResult asyncResult);
// 비동기: 시키는 방식과 결과를 취하는 방식이 다름
// 문제: 매개변수가 많아짐, 받을 때도 복잡하고, EndRead를 언제 호출해야 할지 애매함

// 좀 더 쉬운 방법?
```

- EAP (Event-based Asynchronous Pattern)

```cs
public void ReadAsync(byte[] buffer, int offset, int count);
public event ReadCompletedEventHandler ReadCompleted;

// 결과를 취하는 방식을 이벤트로
// 문제: APM, EAP 둘 다 작업을 시키는 위치와 받는 위치가 다름
// (호출하는 쪽과 결과를 받아 처리하는 함수(콜백) 사이에 컨텍스트를 넘기기 위해 지역변수나 매개변수를 만들어 넘겨야 함)
```

- TAP (Task-based Asynchronous Pattern)

```cs
public Task<int> ReadAsync(byte[] buffer, int offset, int count);
```

Sync, TAP 간의 메서드 원형이 가장 유사함.  
-> 가장 직관적이고, Sync 방식과 닮아 있어 사용하기 쉬운 비동기 패턴  

### Async/Await

내부 동작  
`async` 메서드는 `Task`를 반환  
`await` 키워드는 `Task`를 받아서 `Task`가 완료될 때까지 대기  
`await` 키워드는 `Task`가 완료되면 `Task`의 결과를 반환  

```cs
// 실수 조심 !!
// 이건 함수가 체인되어 있어서, 첫번째 함수가 끝나야 두번째 함수가 실행됨
await SomeMethodAsync();
await SomeMethodAsync();

// ConfigureAwait(false)를 쓰면 첫번째 함수가 끝나기 전에 두번째 함수가 실행됨
await SomeMethodAsync().ConfigureAwait(false);
await SomeMethodAsync().ConfigureAwait(false);

// 아니면 Task를 먼저 받아서, 그걸 await
Task t = SomeMethodAsync();
Task t2 = SomeMethodAsync();
await t;
await t2;

// WhenAll을 쓸 수도 있음
await Task.WhenAll(t, t2);

// WhenAny를 쓸 수도 있음
var tasks = new List<Task>() { t, t2 };
while (tasks.Count > 0)
{
    Task t = await Task.WhenAny(tasks);
    tasks.Remove(t);
}
```

## CancellationTokenSource, CancellationToken

---

- 비동기 작업을 취소하는 용도

- `CancellationTokenSource` 클래스 (cts)
  - `CancellationToken` 을 생성하고, Cancel 요청을 `CancellationToken` 에게 보내는 역할
  - `Cancel()`: 발행한 모든 토큰에 취소 신호
  - `CancelAfter(TimeSpan delay)`: "
  - `IsCancellationRequested`: 취소가 요청되었는지 여부

- `CancellationToken` 구조체
  - 현재 Cancel 상태를 모니터링 하는 구조체
  - 여러 Listener들에 의해 사용됨
  - `Register(Action callback)`: 취소가 요청되었을 때 호출될 콜백을 등록
  - `IsCancellationRequested`: 취소가 요청되었는지 여부

### 사용

1. `CancellationTokenSource` 필드 선언
2. `CancellationTokenSource` 객체 생성
3. 비동기 작업 메서드 안에서 작업이 취소되었는지를 체크하는 코드
   - `if (cancelTokenSource.Token.IsCancellationRequested) => return null;`
4. 취소 요청
   - `cancelTokenSource.Cancel();`
5. `cts.Dispose()`
   - `using` 블록을 썼다면, 블록을 벗어날 때 자동으로 호출
   - 그게 아니라면 임의로 호출

## 메모

---

### 메모

밑 단계는 똑같지만  
조금 추상화해보면  
스레드는 포그라운드 스레드와 백그라운드 스레드로 나뉨  
포그라운드 스레드가 종료되지 않으면 프로세스가 종료되지 않음  
백그라운드 스레드는 프로세스가 종료되면 종료됨  

Delegate: 내부적으로 오브젝팅을 합니다만, 오브젝트안에 콜백 함수에 관한 포인터를 가지고 있는 타입  

비트레벨 parallel  
데이터 parallel  
task parallel  

Task.Delay(1000);
Task.Run(() => { });

async 키워드를 쓰면  
함수가 상태머신으로 변환됨  

상태머신은 함수가 실행되는 동안 상태를 저장하고, 다시 실행될 때 상태를 복원하는 것  

### 참고

- [참고: 'C#을 이용한 Task 병렬화와 비동기 패턴'](https://youtu.be/ZUqUlZ3GjlA)  
- [참고: 'C# 비동기 사용 예제(Task, WhenAll, WhenAny)'](https://youtu.be/44x5KsInMYw)  
- [참고: 'C# 비동기/대기/작업 설명(심층 분석)'](https://youtu.be/il9gl8MH17s)  
- [참고: 'C# - CancellationToken'](https://bacha.tistory.com/137)  

## 동기/비동기

---

### 동기

Synchrounous  

메소드를 호출한 이후, 메소드가 종료될 때까지 코드 실행이 차단됨  
즉, 작업은 순서대로 실행되므로 하나의 작업이 끝나야 다음 작업을 수행할 수 있음  

### 비동기

Asynchronous  

메소드를 호출한 이후 해당 메서드가 종료되지 않아도 코드 기다리지 않고 다음 코드 실행  
비동기 메서드는 백그라운드 스레드에서 수행되므로, 메인스레드는 다른 작업을 수행할 수 있음  
