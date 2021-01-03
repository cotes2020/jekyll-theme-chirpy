---
title: Grand Central Dispatch(GCD)
date: 2020-12-23 10:43:59 +0800
author: grohong
categories: [Concurrency]
tags: [iOS, GCD, Concurrency]
---

## **Threads**

**Thread**는 정확히는 **thread of execution**은 줄임말 입니다. 한 **task(작업)**의 자원을 시스템에서 어떻게 실행시키는지를 뜻하고 있죠. 실제로 우리가 사용하는 **app**에선 여러 **작업(multiple tasks)** 이 **multiple thread**로 동작합니다.\\
이런 **multithreading**로 작업했을 경우 많은 장점이 있습니다.

* **Faster execution**: *concurrently*하게 작업되기 때문에 빠르게 작업이 가능
* **Responsiveness**: UI는 main thread에서만 작업 되기 때문에, 여러 작업을 핟더라도 app 반응에 영향이 없다. 
* **Optimized resource consumption**: OS에 최적화 되어 있다.

하지만, 아래와 같은 짤과 같이 우리가 직접 *OS의 thread를 **직접** 만들어서 사용하게 된다면* 성능은 다 안좋아 지는걸 느낄 수 있습니다.

![thread](/assets/GCD/Thread.gif)
  > 제목: 내가 만든 multi thread

<br>

## **Dispatch queues**

위와 같은 문제를 해결하기 위해 **DispatchQueue**를 사용할 수 있습니다.\\
**DispatchQueue**는 직접 <ins>*thread를 만드는 것이 아닌* OS에게 thread를 만들 queue를 제공하는 것</ins>입니다. 간단하게 말하면 OS에 직접 thread를 만드는 것이 아니고 만들어야할 thread를 queue에 넣어 OS에서 thread를 최적화해서 만들 도록 하는 것입니다.

코드는 간단합니다.

```swift
let label = "com.grohong.dispatch"
let queue = DispatchQueue(label: label)
```
이렇게 **DispatchQueue**에 간단한 라벨링으로 생성한다면, 후에 해당 라벨로 **queue**를 재사용 할 수 있습니다.

<br>

### The main queue

**App** 실행하게 되면 **main dispatch queue**가 만들어 집니다. 해당 <ins>queue의 **serial queue**</ins>에서는 UI 작업이 순차적으로 일어납니다. 그러니 해당 queue에 작업을 넣으면 사용자의 UI 반응에 영향이 가기 때문에, 성능이 저하되는 걸 체감할 수 있습니다.

  > DispatchQueue.main.sync { } 작업은 하면 안된다!

<br>

### Quality of service

**DispatchQueue**는 **serial** 또는 **concurrent**하게 생성할 수 있습니다.\\
**concurrent**한 DispatchQueue도 간단하게 생성 할 수 있습니다.

```swift
let label = "com.grohong.dispatch.concurrent"
let queue = DispatchQueue(label: label, attributes: .concurrent)
```

이때, **DispatchQueue**의 **Quality of service(QoS)**를 설정하여 <ins>우선순위</ins>를 정할 수 있습니다.\\
UI와 상관없는 작업을 **우선순위**를 정하여 사용하고 싶은경우 아래와 같이 **global queue**에 **QoS**를 만들어 넣어주면 됩니다.

  > **Global queues**는 항상 concurrent 이고 FIFO(first in, firsst out) 입니다.

```swift
let queue = DispatchQueue.global(qos: .userInteractive)
```
개별 **DispatchQueue**도 다음과 같이 간단하게 만들 수 있습니다.

```swift
let queue = DispatchQueue(label: label, 
                          qos: .userInitiated,
                          attributes: .concurrent)
```

**QoS**는 6가지로 나뉘게 됩니다.

#### .userInteractive

**userInteractive**는 가장 높은 순위로 UI 반응과 동시에 빠르게 반응할 작업을 넣어줍니다.
<br>

#### .userInitiated

**.userInitiated**는 사용자의 반응과 동시에 시작하지만, 비동기적으로 처리될 작업을 넣어줍니다.
<br>

#### .utility

위의 QoS는 성능이 중요한 작업이라면,
**.utility**는 I/O, networking 과 data feed 등 급하지 않은 일을 에너지 효율과 성능을 동시에 신경 쓰는 작업을 넣어줍니다.
<br>

#### .background

**.background**는 사용자가 눈치채지 못할 정도의 작업을 넣어 줍니다.
<br>

#### .default and .unspecified

우선순위를 안정해준 **default value**이고, **unspecified** 거의 사용하지 않는 단계의 작업을 뜻합니다.

<br>

### Adding task to queues

**Dispatch queues**는 **task**에 **method**를 **sync** 또는 **async** 하게 처리하도록 추가할 수 있습니다.\\
간단한 **async** **method**를 추가하는 코드를 봐보겠습니다.

```swift
DispatchQueue.global(qos: .utility).async { [weak self] in
  guard let self = self else { return }

  // Perform your work here
  // ...

  // Switch back to the main queue to
  // update your UI
  DispatchQueue.main.async {
    self.textLabel.text = "New articles available!"
  }
}
```

여기서 주의할 점은 <ins>UI의 업데이트는 항상 **main queue**에서 이뤄져야</ins> 한다는 것입니다.

<br>

## **DispatchGroup**

**DispatchGroup**은 **class**로 각각의 **queue**의 **tasks**를 **group**으로 묶어 관리할 수 있습니다.

```swift
let group = DispatchGroup()

someQueue.async(group: group) { ... your work ... } 
someQueue.async(group: group) { ... more work .... }
someOtherQueue.async(group: group) { ... other work ... } 

group.notify(queue: DispatchQueue.main) { [weak self] in
  self?.textLabel.text = "All jobs have completed"
}
```

위에 코드에서 볼 수 있듯이, 각각의  **queue**에 **group**을 설정 할 수 있습니다.\\
**DispatchGroups**은 ```notify(queue: )```제공합니다. 해당 함수는 **group**에 모든 **task**가 모드 끝났을때 알려줍니다.

<br>

### Synchronous waiting

만약 **DispatchGroup**에서 **task**가 끝나지 않아 ```notify(queue: )```를 받지 못하는 경우가 생길 수 있습니다.\\
 이럴 경우 ```wait(timeout)``` method를 사용하여 ```.timeOut``` 을 설정할 수 있습니다.

```swift
let group = DispatchGroup()
let queue = DispatchQueue.global(qos: .userInitiated)

queue.async(group: group) {
  print("Start job 1")
  Thread.sleep(until: Date().addingTimeInterval(10))
  print("End job 1")
}

queue.async(group: group) {
  print("Start job 2")
  Thread.sleep(until: Date().addingTimeInterval(2))
  print("End job 2")
}

if group.wait(timeout: .now() + 5) == .timedOut {
  print("I got tired of waiting")
} else {
  print("All the jobs have completed")
}
```

위 코드를 보면, ```wait(timeout)```에 5초의 ```timeOut```를 주어 만약 **timeout** 파라미터 시간내에 **task**가 끝나지 않는다면 조건문에서 예외처리가 가능합니다.

<br>

### Wrapping asynchronous methods

**DispatchGroup**에서는 위에 예제 처럼 **queue**의 **task**가 끝날경우 **group**에 **noti**를 줍니다. 하지만 **queue**의 **task** 내부의 작업이 **async**하게 끝날 경우에는 **task**가 다 끝나지 않은 상황에서 **noti**가 가는 경우가 생길 것입니다.\\
이럴때 **DispatchGroup**의 **enter** 와 **leave**를 이용하여 해결 할 수 있습니다.

```swift
queue.dispatch(group: group) {
  // count is 1
  group.enter()
  // count is 2
  someAsyncMethod { 
    defer { group.leave() }
    
    // Perform your work here,
    // count goes back to 1 once complete
  }
}
```

위의 코드 처럼 **task**가 시작될때 **enter()**를 호출하여 **DispatchGroup**에 들어가는 것을 알리고, **async**한 **task**가 끝날 경우 **defer**를 이용하여 마지막에 **task**가 마무리 됐음을 **leave()** 알려주면 됩니다.

<br>

## **Semaphores**

**Semaphore**을 이용하면 **task**에 사용되는 리소스를 관리할 수 있습니다.\\
아래 코드 처럼 **DispatchSemaphore**를 생성할때 사용할 리소를 정할 수 있습니다.

```swift
let semaphore = DispatchSemaphore(value: 4)
```
해당 **DispatchSemaphore**는 **wait**를 이용해 리소스를 준비하고, **signal**를 이용하여 리소스 해제 타임을 조절할 수 있습니다.

```swift
let group = DispatchGroup()
let queue = DispatchQueue.global(qos: .userInteractive)
let semaphore = DispatchSemaphore(value: 1)

let base = "https://wolverine.raywenderlich.com/books/con/image-from-rawpixel-id-"
let ids = [466881, 466910, 466925, 466931, 466978, 467028, 467032, 467042, 467052]

var images: [UIImage] = []

for id in ids {
    guard let url = URL(string: "\(base)\(id)-jpeg.jpg") else { continue }

    semaphore.wait()
    group.enter()

    let task = URLSession.shared.dataTask(with: url) { data, _, error in
        defer {
            print("Finish download \(id)")
            group.leave()
            semaphore.signal()
        }

        if error == nil,
           let data = data,
           let image = UIImage(data: data) {
            images.append(image)
        }
    }

    task.resume()
}

// Finish download 466881
// Finish download 466910
// Finish download 466925
// Finish download 466931
// Finish download 466978
// Finish download 467028
// Finish download 467032
// Finish download 467042
// Finish download 467052
```

위의 예제를 본다면, **DispatchSemaphore**의 리소스가 하나이기 때문에 이미지가 download가 순차적으로 이뤄지는걸 확인 할 수 있습니다.\\
하지만 리소스를 4개로 늘렸을 경우 리소스의 **task**가 끝나는 타이밍에 따라 다운로드가 다운이 이루어 집니다.

```swift
let semaphore = DispatchSemaphore(value: 4)

// Finish download 466881
// Finish download 466910
// Finish download 466931
// Finish download 467028
// Finish download 467032
// Finish download 467042
// Finish download 466925
// Finish download 466978
// Finish download 467052

// Always change....
```