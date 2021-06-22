---
layout: post
title: "Alamofire 源码剖析"
date: 2020-10-10 22:42:00.000000000 +09:00
categories: [Swift]
tags: [Swift, Alamofire]
---

大部分App都需要跟服务端进行数据交互，那么App是怎样进行数据交互的呢？通常情况下都是通过 HTTPS/HTTP 来完成的，`Swift`为我们封装一个`URLSession`非常实用的的类来进行数据通信，但是对于比较复杂的网络请求，我们还需要把网络层跟业务层分离开来，这样就必须对`URLSession`进行再一起封装，对 method ，header ，上传，下载，错误处理等再做一层处理。那么本文介绍的一个框架`Alamofire`。

## 前言

在功能角度，`Alamofire`是一个http网络请求框架，在代码实现的角度上看`Alamofire`是对`URLSession`的封装。在分析`Alamofire`之前先看看`URLSession`。

`URLSession `发起网络请求

```swift
// URLSession
let session = URLSession.shared
let dataTask = session.dataTask(with: URL(string: "https://jovins.cn/")!) { (data, response, error) in
  guard let data = data,
    let string = String(data: data, encoding: .utf8) else {
    return
  }
  print(string)
}
dataTask.resume()
```

`Alamofire` 发起网络请求

```swift
// Alamofire
Alamofire.request("https://jovins.cn/")
  .responseString { (response) in
    guard let string = response.value else { return }
    print(string)
}
```

两者比较可以看出`Alamofire`简洁，不需要再初始化一个`URLSessionDataTask`和调用resume，也不需要对请求结构进行解码和格式转换，而且对网络请求的错误处理、参数转换和结果处理，`Alamofire`已经提供了一系列丰富的链式调用的处理方法，代码简洁、比较容易理解。

## Alamofire框架结构

### 目录结构

首先看一下 `Alamofire`中的目录结构, 来了解一下它是如何组织各个文件的.

```swift
-- Alamofire.swift
	-- Core
		-- AFError.swift
		-- HTTPHeaders.swift
		-- HTTPMethod.swift
		-- Notifications.swift
		-- ParameterEncoder.swift
		-- ParameterEncoding.swift
		-- Protected.swift
		-- Request.swift
		-- RequestTaskMap.swift
		-- Response.swift
		-- Session.swift
		-- SessionDelegate.swift
		-- URLConvertible+URLRequestConvertible.swift
	-- Extensions
		-- DispatchQueue+Alamofire.swift
		-- OperationQueue+Alamofire.swift
		-- Result+Alamofire.swift
		-- StringEncoding+Alamofire.swift
		-- URLRequest+Alamofire.swift
		-- URLSessionConfiguration+Alamofire.swift
	-- Features
		-- AlamofireExtended.swift
		-- AuthenticationInterceptor.swift
		-- CachedResponseHandler.swift
		-- Combine.swift
		-- EventMonitor.swift
		-- MultipartFormData.swift
		-- MultipartUpload.swift
		-- NetworkReachabilityManager.swift
		-- RedirectHandler.swift
		-- RequestInterceptor.swift
		-- ResponseSerialization.swift
		-- RetryPolicy.swift
		-- ServerTrustEvaluation.swift
		-- URLEncodedFormEncoder.swift
		-- Validation.swift
```

+ `Core`里面包含的是网络请求过程中必须要调用的部分
+ `Extensions`为系统的一些类添加了便捷方法
+ `Features`则是`Alamofire` 提供的一些特定功能，如缓存策略，请求结果的处理等

### 框架架构

`Alamofire` 基本架构如下图所示：

![](/assets/images/swift-alamofire-01.png)

整个Alamofire的设计非常简洁，其中定义的类除了需要进行缓存的请求，会话是使用class类型，其余几乎都是采用协议与结构体实现。完整的将网络请求，URL校验，失败重试机制，请求缓存，错误机制，响应解析等功能。

## 源码解析

### Session 

`Session`类是`Alamofire`的核心类, 封装了`URLSession`类, 管理所有的网络请求调度.

```swift
/// 一个默认的单例
public static let `default` = Session()
/// 持有一个URLSession, 用来创建请求Task, 注意不能跟本对象持有的URLSessionTask产生交互， 否则会影响Alamofire内部逻辑
public let session: URLSession
/// 处理URLSession代理, URLSessionTaskDelegate, 以及请求拦截等逻辑
public let delegate: SessionDelegate
/// 内部回调执行以及状态更新的队列，必须是串行队列
public let rootQueue: DispatchQueue
/// 是否在Request创建的时候就立刻发送, 该属性用来统一管理Request创建时的startImmediately参数, 默认true
public let startRequestsImmediately: Bool
/// 异步创建Request的队列，默认是rootQueue
public let requestQueue: DispatchQueue
/// 解析response的队列, 默认是rootQueue
public let serializationQueue: DispatchQueue
/// 请求拦截器接口，是RequestAdapter跟RequestRetrier的结合, 默认nil
public let interceptor: RequestInterceptor?
/// 证书信任器接口, 默认nil
public let serverTrustManager: ServerTrustManager?
/// 重定向处理器接口, 默认nil
public let redirectHandler: RedirectHandler?
/// 缓存管理接口, 默认nil
public let cachedResponseHandler: CachedResponseHandler?
/// 事件监测管理器类，处理请求生命周期各阶段的事件, 默认用下面的defaultEventMonitors以及传入的事件检测器初始化
public let eventMonitor: CompositeEventMonitor
/// 默认的事件监测器接口列表，只有一个通知事件监测器
public let defaultEventMonitors: [EventMonitor] = [AlamofireNotifications()]
/// 结构体, 用来保存Request跟URLSessiontask映射关系, 提供了各种方法来存取task跟request以及数量判断, 为空判断
var requestTaskMap = RequestTaskMap()
/// 当前正在请求的Request集合
var activeRequests: Set<Request> = []
/// 等待成功的回调
var waitingCompletions: [URLSessionTask: () -> Void] = [:]
```

**1.Session初始化**

`Session`有两个初始化方法, 一个必要初始化方法, 一个便捷初始化方法.

```swift
// 通过URLSession和parameters创建Session
public init(session: URLSession,
                delegate: SessionDelegate,
                rootQueue: DispatchQueue,
                startRequestsImmediately: Bool = true,
                requestQueue: DispatchQueue? = nil,
                serializationQueue: DispatchQueue? = nil,
                interceptor: RequestInterceptor? = nil,
                serverTrustManager: ServerTrustManager? = nil,
                redirectHandler: RedirectHandler? = nil,
                cachedResponseHandler: CachedResponseHandler? = nil,
                eventMonitors: [EventMonitor] = []) {
  ...
}
```

```swift
/// 通过URLSessionConfiguration配置创建Session
public convenience init(configuration: URLSessionConfiguration = URLSessionConfiguration.af.default,
                            delegate: SessionDelegate = SessionDelegate(),
                            rootQueue: DispatchQueue = DispatchQueue(label: "org.alamofire.session.rootQueue"),
                            startRequestsImmediately: Bool = true,
                            requestQueue: DispatchQueue? = nil,
                            serializationQueue: DispatchQueue? = nil,
                            interceptor: RequestInterceptor? = nil,
                            serverTrustManager: ServerTrustManager? = nil,
                            redirectHandler: RedirectHandler? = nil,
                            cachedResponseHandler: CachedResponseHandler? = nil,
                            eventMonitors: [EventMonitor] = []) {
  
}
```

**2.管理所有请求API**

```swift
/// 对所有正在执行的请求执行一个闭包
public func withAllRequests(perform action: @escaping (Set<Request>) -> Void) {
    rootQueue.async {
       action(self.activeRequests)
    }
}
/// 取消全部请求
public func cancelAllRequests(completingOnQueue queue: DispatchQueue = .main, completion: (() -> Void)? = nil) {
   withAllRequests { requests in
     requests.forEach { $0.cancel() }
       queue.async {
           completion?()
       }
   }
}
```

**3.初始化数据请求**

```swift
/// 用来转换URLRequest对象的闭包
public typealias RequestModifier = (inout URLRequest) throws -> Void
/// 普通的request转换器, 使用ParameterEncoding协议对象来编码参数
struct RequestConvertible: URLRequestConvertible {
    let url: URLConvertible 				/// url
    let method: HTTPMethod 					/// 请求方法
    let parameters: Parameters?			/// 参数
    let encoding: ParameterEncoding	/// 参数编码对象，默认url编码
    let headers: HTTPHeaders?				/// 请求头
    let requestModifier: RequestModifier?
}
/// 参数符合Encodable协议的转换器, 使用ParameterEncoder协议对象编码参数
struct RequestEncodableConvertible<Parameters: Encodable>: URLRequestConvertible {
    let url: URLConvertible
    let method: HTTPMethod
    let parameters: Parameters?
    let encoder: ParameterEncoder
    let headers: HTTPHeaders?
    let requestModifier: RequestModifier?
}
/// 通过URLRequestConvertible的值和RequestInterceptor适配器创建DataRequest
open func request(_ convertible: URLRequestConvertible, interceptor: RequestInterceptor? = nil) -> DataRequest {
    let request = DataRequest(convertible: convertible,
                                  underlyingQueue: rootQueue,
                                  serializationQueue: serializationQueue,
                                  eventMonitor: eventMonitor,
                                  interceptor: interceptor,
                                  delegate: self)

    perform(request)
    return request
}
```

**4.准备发送请求**

```swift
func perform(_ request: Request) {
    rootQueue.async {
      	/// 先在rootQueue中判断是否请求被取消
        guard !request.isCancelled else { return }
        /// 将request塞入正在请求Request集合中
        self.activeRequests.insert(request)
      	/// 在requestQueue队列发送请求
        self.requestQueue.async {
            /// 子类必须先匹配，否则就会被识别为父类了
            switch request {
            /// UploadRequest是DataRequest的子类,优先匹配
            case let r as UploadRequest: self.performUploadRequest(r)
            /// 1.
            case let r as DataRequest: self.performDataRequest(r)
            case let r as DownloadRequest: self.performDownloadRequest(r)
            /// 2.
            case let r as DataStreamRequest: self.performDataStreamRequest(r)
            default: fatalError("Attempted to perform unsupported Request subclass: \(type(of: request))")
            }
        }
    }
}
```

+ 1.通过`rootQueue`告知事件监听器`didCreateUploadable`开始调用performSetupOperations方法。如果创建失败则在`rootQueue`告知监视器`didFailToCreateUploadable`。
+ 2.如果是新建下载，直接调用`performSetupOperations`方法；如果是断点续传，在`rootQueue`调用`didReceiveResumeData`方法。

```swift
func performSetupOperations(for request: Request,
                                convertible: URLRequestConvertible,
                                shouldCreateTask: @escaping () -> Bool = { true })
    {
    /// 当前在requestQueue  
		dispatchPrecondition(condition: .onQueue(requestQueue))
    let initialRequest: URLRequest
    do {
        /// 检测请求是否有效
        initialRequest = try convertible.asURLRequest()
        try initialRequest.validate()
    } catch {
      	/// 出错的话就在rootQueue队列上报错误
        rootQueue.async { request.didFailToCreateURLRequest(with: error.asAFError(or: .createURLRequestFailed(error: error))) }
        return
    }
		/// 通过rootQueue通知request初始化URLRequest成功
    rootQueue.async { request.didCreateInitialURLRequest(initialRequest) }
    /// 判断请求是否取消
    guard !request.isCancelled else { return }
    /// 检查是否有请求适配器
    /// 判断request的拦截器跟Session的拦截器都不为空的话, 就返回生成组合拦截器
    /// 返回request拦截器或者Session拦截器
    guard let adapter = adapter(for: request) else {
        guard shouldCreateTask() else { return }
      	/// 没有拦截器的话直接通知
        rootQueue.async {
          self.didCreateURLRequest(initialRequest, for: request) 				 }
        return
    }
    /// 使用拦截器中的适配器来预处理请求
    adapter.adapt(initialRequest, for: self) { result in
        do {
            /// 通知处理完成
            let adaptedRequest = try result.get()
            try adaptedRequest.validate()
            self.rootQueue.async { request.didAdaptInitialRequest(initialRequest, to: adaptedRequest) }
            guard shouldCreateTask() else { return }
            self.rootQueue.async { self.didCreateURLRequest(adaptedRequest, for: request) }
        } catch {
           /// 抛出requestAdaptationFailed错误
           self.rootQueue.async { request.didFailToAdaptURLRequest(initialRequest, withError: .requestAdaptationFailed(error: error)) }
        }
    }
}
```

方法`performSetupOperations`接受两个参数: `Request`对象以及`URLRequestConvertible`协议对象, 后者来自于`request.convertible`属性. 处理成功后, 会调用`didCreateURLRequest`方法来更新状态。

**5.创建`URLSessionTask`发送请求**

当创建请求完成拦截适配器处理完成之后, 就会来到这里的逻辑, 除了断点续传的请求会执行`didReceiveResumeData`方法, 其他的几个请求都会执行`didCreateURLRequest`方法, 然后最终都会调用`updateStatesForTask`方法来更新。

```swift
func didCreateURLRequest(_ urlRequest: URLRequest, for request: Request) {
    dispatchPrecondition(condition: .onQueue(rootQueue))
    request.didCreateURLRequest(urlRequest)
    guard !request.isCancelled else { return }
    let task = request.task(for: urlRequest, using: session)
    requestTaskMap[request] = task
    request.didCreateTask(task)
    updateStatesForTask(task, request: request)
}
```

基本逻辑跟上面相似, 区别就是创建task的方法不同, 使用已下载`Data`创建`URLSessionDownloadTask`。

```swift
func didReceiveResumeData(_ data: Data, for request: DownloadRequest) {
    dispatchPrecondition(condition: .onQueue(rootQueue))
    guard !request.isCancelled else { return }
    let task = request.task(forResumeData: data, using: session)
    requestTaskMap[request] = task
    request.didCreateTask(task)
		updateStatesForTask(task, request: request)
}
```

开始发送请求。

```swift
func updateStatesForTask(_ task: URLSessionTask, request: Request) {
    dispatchPrecondition(condition: .onQueue(rootQueue))
    request.withState { state in
        switch state {
        case .initialized, .finished:
            /// 初始化或者请求完成, Do nothing.
            break
        case .resumed:
            /// 发送请求
            task.resume()
            rootQueue.async { request.didResumeTask(task) }
        case .suspended:
            /// 请求挂起
            task.suspend()
            rootQueue.async { request.didSuspendTask(task) }
        case .cancelled:
            /// 先恢复task 再取消, 保证task被取消
            task.resume()
            task.cancel()
            rootQueue.async { request.didCancelTask(task) }
        }
    }
}
```

`Session`类的作用：

+ 初始化并管理`URLSession`， 持有`SessionDelegate`对象管理`URLSession`的相关代理方法。
+ 根据不同的参数， 派发到不同方法创建不同的`Request`子类对象，使用拦截适配器预处理`Request`对象，调用`Request`的方法创建`URLSessionTask`子类，将`Request`与`URLSessionTask`链接配对，调用resume发送task请求。
+ 实现`RequestDelegate`协议提供创建`Request`对象时的两个控制参数，以及处理重试逻辑，实现`SessionStateProvider`协议处理`Request`跟`Task`的状态。

### Request

`Request`类是`Alamofire`中对`URLRequest`以及`URLSessionTask`的封装，每次发起网络请求都会生成一个对应的子类对象，根据不同的网络请求会生成 `DataRequest` ，`DataStreamRequest` ， `DownloadRequest` 和 `UploadRequest`。

**`Request`基类State状态**

```swift
// 处理状态之间的转换。
public enum State {
    /// 初始化状态
    case initialized
    /// 调用resume()时会更新resumed状态，同时也会调用对应的task的 resume()方法
    case resumed
    /// 调用suspend()时会更新为suspended状态，同时也会调用对应的Task 的suspend()方法
    case suspended
    /// 调用cancel()时会更新为cancelled状态，同时也会调用对应的task的 cancel()方法
    /// 跟 resumed 和 suspended 状态不同，转换至 cancelled 状态后无法再转换至其它状态
    case cancelled
    /// 完成所有请求结果的序列化操作
    case finished
    /// 判断当前状态是否可以转换到其它状态
    func canTransitionTo(_ state: State) -> Bool {
		...
    }
}
```

**`MutableState`**

`Request`定义了一个结构体`MutableState`，用于将可变属性和不可变属性区分开，所有可变属性到放到 `MutableState` 中，且使用 `@Protected` 声明来保证修改这些属性时是线程安全的。

**`Request` 的不可变属性**

```swift
/// UUID作为Request的唯一id使用，用于支持Hashble和Equatable协议
public let id: UUID
/// 串行队列，Request内所有异步调用都是在这个队列内进行
public let underlyingQueue: DispatchQueue
/// 进行序列化时所使用到的队列，默认跟underlyingQueue一致
public let serializationQueue: DispatchQueue
/// EventMonitor，下面会说到具体的作用
public let eventMonitor: EventMonitor?
/// Request's interceptor ，用于URL适配，请求重试，认证等
public let interceptor: RequestInterceptor?
/// Request's delegate ，用于调用Session的一些方法
public private(set) weak var delegate: RequestDelegate?
```

**`RequestDelegate`**

`RequestDelegate` 负责提供一些 `Session` 的方法给 `Request` 调用， `Request` 的 `delegate` 都是 `Session` 对象。

```swift
public protocol RequestDelegate: AnyObject {
    /// 创建URLSessionTask时使用到的URLSessionConfiguration
    var sessionConfiguration: URLSessionConfiguration { get }
    /// 是否直接开始，如果startImmediately为true，则当添加第一个 response handler 时就会调用resume()方法发送请求
    var startImmediately: Bool { get }
    /// 清除Session中关于Request的记录
    func cleanup(after request: Request)
    /// 异步调用delegate的方法，用于判断Request是否需要进行重试
    func retryResult(for request: Request, dueTo error: AFError, completion: @escaping (RetryResult) -> Void)
    /// 异步重试 Request
    func retryRequest(_ request: Request, withDelay timeDelay: TimeInterval?)
}
```

**`RequestTaskMap`**

`RequestTaskMap` 用于串联 `URLSessionTask` 和 `Request` ，当请求结果更新时，需要根据 `URLSessionTask` 找到对应的 `Request` ，进行处理。或者通过 `Request` 找到对应的 `URLSessionTask` 。定义了两个 `Dictionary` ，用于实现两者之间的 `map` ，通过 `subscript` 方法提供下标设置和访问。

```swift
struct RequestTaskMap {
    private var tasksToRequests: [URLSessionTask: Request]
    private var requestsToTasks: [Request: URLSessionTask]
	  init(tasksToRequests: [URLSessionTask: Request] = [:],
         requestsToTasks: [Request: URLSessionTask] = [:],
         taskEvents: [URLSessionTask: (completed: Bool, metricsGathered: Bool)] = [:]) {
        self.tasksToRequests = tasksToRequests
        self.requestsToTasks = requestsToTasks
        self.taskEvents = taskEvents
    }

	  subscript(_ request: Request) -> URLSessionTask? {
       ...
    }

    subscript(_ task: URLSessionTask) -> Request? {
        ...
    }
}
```

**`Request`四个子类**

+ `DataRequest`: 使用`URLSessionDataTask`处理请求数据, 数据保存在内存中, 使用Data对象储存。适合普通请求, 小图片请求, 小文件请求。
+ `DataStreamRequest`: 处理流式请求，使用`OutputStream`传出数据。
+ `DownloadRequest`: 使用`URLSessionDownloadTask`处理下载请求，文件下载在磁盘上。
+ `UploadRequest`: 使用`URLSessionUploadTask`处理上传请求，上传内容可以用表单、文件、 InputStream。

**`Request`开放相关方法**

```swift
// MARK: - 开放apis, 可以被module外部调用, 可以在任何队列调用
// MARK: 状态相关
/// 取消Request, 一旦取消了, 就再也不能继续跟挂起了
@discardableResult
public func cancel() -> Self {
    ...
    return self
}
/// 挂起请求
@discardableResult
public func suspend() -> Self {
    $mutableState.write { mutableState in
        /// 检测是否能挂起
        guard mutableState.state.canTransitionTo(.suspended) else { return }
        /// 更新状态
        mutableState.state = .suspended
        /// 通知调用suspend()完成
        underlyingQueue.async { self.didSuspend() }
        /// 检测最后一个请求是否完成, 完成的话跳过
        guard let task = mutableState.tasks.last, task.state != .completed else { return }
        /// 挂起task
        task.suspend()
        /// 通知task挂起完成
        underlyingQueue.async { self.didSuspendTask(task) }
    }
    return self
}
/// 继续task
@discardableResult
public func resume() -> Self {
    $mutableState.write { mutableState in
    /// 检查能否继续
    guard mutableState.state.canTransitionTo(.resumed) else { return }
        /// 更新状态
        mutableState.state = .resumed
        /// 通知调用resume()完成
        underlyingQueue.async { self.didResume() }
        /// 检测task是否全部完成
        guard let task = mutableState.tasks.last, task.state != .completed else { return }
        /// 继续task
        task.resume()
        ///通知task继续完成
        underlyingQueue.async { self.didResumeTask(task) }
    }
    return self
}
```

**`Request类扩展相关`**

扩展`Request`对象来实现协议

```swift
extension Request: Equatable {
    //Equatable协议用uuid来比较
    public static func ==(lhs: Request, rhs: Request) -> Bool {
        lhs.id == rhs.id
    }
}
extension Request: Hashable {
    //Hashable用uuid做key
    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}
/// 对Request对象的文本描述, 包括请求方法, url, 如果已经有响应数据, 会加上响应状态码
extension Request: CustomStringConvertible {
 ... 
}
/// 扩展Request, 返回请求的curl格式(应该是linux环境下调试用)
extension Request {
    public func cURLDescription() -> String {
      ...
    }
}
```

`Request`类主要负责连接`URLRequest`与`URLSessionTask`， 基类主要是定义了一大堆请求中用到的属性， 然后定义了请求中各个阶段的回调方法， 给外部或者自己调用来通知监听器。

### RequestInterceptor拦截器

`RequestInterceptor`请求拦截器是一个协议，用来在请求流程中拦截请求，并对请求进行一些必要的处理，这是一个组合协议，`RequestAdapter`请求适配器与`RequestRetrier`请求重试器。使用者可以自己实现请求拦截器，根据自己的需求适配`URLRequest`，或者定义自己的重试逻辑。

**1.`RequestAdapter`请求适配器**

```swift
/// 入参为初始URLRequest, Session以及适配完成后的回调, 回调参数为Result对象, 可以为成功适配后的URLRequest对象, 也可以返回错误, 会向上抛出从创建requestAdaptationFailed错误
func adapt(_ urlRequest: URLRequest, for session: Session, completion: @escaping (Result<URLRequest, Error>) -> Void)
```

主要用来在`Request`创建好初始的`URLRequest`后，对`URLReques`t进行适配，适配处理前后均会告诉监听器对应的通知。

**2.`RequestRetrier`请求重试器**

```swift
// 参数为: Request对象, Session, 请求失败的错误信息以及重试逻辑回调, 回调参数为重试逻辑, 调用者根据该逻辑决定重试行为
func retry(_ request: Request, for session: Session, dueTo error: Error, completion: @escaping (RetryResult) -> Void)
```

主要用来在请求失败时，决定是直接抛出错误，还是依据一定的逻辑进行重试。

**3.`RetryResult`重试逻辑**

```swift
public enum RetryResult {
    /// 立刻重试
    case retry
    /// 延迟重试
    case retryWithDelay(TimeInterval)
    /// 不重试,直接完成请求
    case doNotRetry
    /// 不重试并抛出错误
    case doNotRetryWithError(Error)
}
//扩展一下快速取得相关信息, 两个可选值属性方便快速做出判断
extension RetryResult {
    /// 是否需要重试
    var retryRequired: Bool {...}
    /// 延迟重试时间
    var delay: TimeInterval? {...}
    /// 不重试并抛出错误时的错误信息
    var error: Error? {...}
}
```

**4.RequestInterceptor组合**

```swift
public protocol RequestInterceptor: RequestAdapter, RequestRetrier {}
//扩展一下，使得即便遵循协议也可以不实现方法，依旧不会报错
extension RequestInterceptor {
    public func adapt(_ urlRequest: URLRequest, for session: Session, completion: @escaping (Result<URLRequest, Error>) -> Void) {
        //直接返回原请求
        completion(.success(urlRequest))
    }
    public func retry(_ request: Request, for session: Session, dueTo error: Error, completion: @escaping (RetryResult) -> Void) 		{
        //不重试
        completion(.doNotRetry)
    }
}
```

**5.`Alamofire`预先定义的请求拦截器**

+ `Adapter`: 基于闭包的简易请求适配器
+ `Retrier`: 基于闭包的简易重试器
+ `Interceptor`: 可以持有多个适配器与重试器

### HTTPMethod

定义了一个结构体来封装`HTTPMethod`，使用一个rawValue来保存字符串，并实现了三个必要协议。

```swift
public struct HTTPMethod: RawRepresentable, Equatable, Hashable {
    /// `CONNECT` method.
    public static let connect = HTTPMethod(rawValue: "CONNECT")
    /// `DELETE` method.
    public static let delete = HTTPMethod(rawValue: "DELETE")
    /// `GET` method.
    public static let get = HTTPMethod(rawValue: "GET")
    /// `HEAD` method.
    public static let head = HTTPMethod(rawValue: "HEAD")
    /// `OPTIONS` method.
    public static let options = HTTPMethod(rawValue: "OPTIONS")
    /// `PATCH` method.
    public static let patch = HTTPMethod(rawValue: "PATCH")
    /// `POST` method.
    public static let post = HTTPMethod(rawValue: "POST")
    /// `PUT` method.
    public static let put = HTTPMethod(rawValue: "PUT")
    /// `TRACE` method.
    public static let trace = HTTPMethod(rawValue: "TRACE")
    public let rawValue: String
    public init(rawValue: String) {
        self.rawValue = rawValue
    }
}
```

### HTTPHeaders

`HTTPHeaders` 是一个 `Struct` 类型，保存了 HTTP 头的一些 name / value 配对。 `HTTPHeader` 则封装了一些快速生成 HTTP 头属性的方法。可以使用 `URLSessionConfiguration.af.default` 来获取默认的 `HTTPHeaders` 对应的 `URLSessionConfiguration` 。 为了方便直接生成 `HTTPHeaders` ， 还支持 `ExpressibleByDictionaryLiteral` 和 `ExpressibleByArrayLiteral` 协议，可以直接使用 `Dictionary` 和 `Array` 的 字面表达式来直接生成。

```swift
/// 扩展HTTPHeaders协议，方便增删改查操作
extension HTTPHeaders: ExpressibleByDictionaryLiteral {
    public init(dictionaryLiteral elements: (String, String)…) {
        self.init()
        elements.forEach { update(name: $0.0, value: $0.1) }
    }
}
extension HTTPHeaders: ExpressibleByArrayLiteral {
    public init(arrayLiteral elements: HTTPHeader…) {
        self.init(elements)
    }
}
// 这里的 .authorization 和 .accpet 都是 HTTPHeader 中为了方便我们调用提供的初始化方法。
let headers: HTTPHeaders = [
    .authorization(username: "Username”, password: “Password"),
    .accept(“application/json”)
]
```

快速创建默认请求头`HTTPHeaders`

```swift
extension HTTPHeaders {
    public static let `default`: HTTPHeaders = [.defaultAcceptEncoding, .defaultAcceptLanguage,.defaultUserAgent]                                             
}
```

对于一些系统的类，也提供了便捷方法来获取 `HTTPHeaders` 

```swift
extension URLRequest {
    public var headers: HTTPHeaders {
        get { allHTTPHeaderFields.map(HTTPHeaders.init) ?? HTTPHeaders() }
        set { allHTTPHeaderFields = newValue.dictionary }
    }
}
extension HTTPURLResponse {
    public var headers: HTTPHeaders {
        (allHeaderFields as? [String: String]).map(HTTPHeaders.init) ?? HTTPHeaders()
    }
}
public extension URLSessionConfiguration {
    var headers: HTTPHeaders {
        get { (httpAdditionalHeaders as? [String: String]).map(HTTPHeaders.init) ?? HTTPHeaders() }
        set { httpAdditionalHeaders = newValue.dictionary }
    }
}
```

### URLConvertible & URLRequestConvertible

`URLconvertible` 协议，用于转化为 `URL`，`URLRequestConvertible`协议定义了可以将任意对象转换为`URLRequest`对象，在创建`Request`对象时使用。这两个协议抽象了`URL`与`URLRequest`，使得创建请求时不再局限于必须使用这两个对象来初始化，可以使用任意符合两个协议的对象即可，方便上层封装解耦。

#### URLConvertible协议

遵循该协议的对象只需要实现一个方法，生成一个URL即可。

```swift
public protocol URLConvertible {
    func asURL() throws -> URL
}
```

`String` 、 `URL` 和 `URLComponents` 默认支持 `URLConvertible` 

```swift
extension String: URLConvertible {
    public func asURL() throws -> URL {
        guard let url = URL(string: self) else { throw AFError.invalidURL(url: self) }
        return url
    }
}
extension URL: URLConvertible {
    public func asURL() throws -> URL { self }
}
extension URLComponents: URLConvertible {
    public func asURL() throws -> URL {
        guard let url = url else { throw AFError.invalidURL(url: self) }
        return url
    }
}
```

`Alamofire`内部定义遵循该协议的对象，`Alamofire`默认支持以下三种发起请求的方式

```swift
/// String
let urlString = "https://jovins.cn/"
Alamofire.request(urlString)
/// URL
let url = URL(string: urlString)!
Alamofire.request(url)
/// URLComponents
let urlComponents = URLComponents(url: url, resolvingAgainstBaseURL: true)!
Alamofire.request(urlComponents)
```

#### URLRequestConvertible协议

类似`URLConvertible`，也是只有一个方法，生成的是`URLRequest`对象。通过 `URLRequest` 可以生成对应的 `URLSessionTask` ，`URLRequest` 默认支持 `URLRequestConvertible`。

```swift
public protocol URLRequestConvertible {
    func asURLRequest() throws -> URLRequest
}
/// 扩展协议，快速获取URLRequest，返回的是个可选值，如果抛出错误，会忽略错误并返回nil：
extension URLRequestConvertible {
    public var urlRequest: URLRequest? { try? asURLRequest() }
}
extension URLRequest: URLRequestConvertible {
    public func asURLRequest() throws -> URLRequest { self }
}
// 给URLRequest添加了一个初始化方法，方便在初始化时直接设置method和allHTTPHeaderFields
extension URLRequest {
    public init(url: URLConvertible, method: HTTPMethod, headers: HTTPHeaders? = nil) throws {
        let url = try url.asURL()
        self.init(url: url)
        httpMethod = method.rawValue
        allHTTPHeaderFields = headers?.dictionary
    }
}
```

通过 `URLRequest` 发起请求：

```swift
let url = URL(string: "https://jovins.cn/")!
var urlRequest = URLRequest(url: url)
urlRequest.method = .post
let parameters = ["favor": "test"]
do {
    urlRequest.httpBody = try JSONEncoder().encode(parameters)
} catch {
    // Handle error.
}
urlRequest.headers.add(.contentType("application/json"))
Alamofire.request(urlRequest)
```

这两个协议，一般是需要对`Alamofire`进行二次封装时会用到，对一些通用数据进行封装，以及一些需要针对业务需求进行变更的数据进行处理。比如下面对`URLRequestConvertible`二次封装:

```swift
enum HJRouter: URLRequestConvertible {
    case get([String: String]), post([String: String])
    var baseURL: URL {
        return URL(string: "https://jovins.cn/")!
    }
    var method: HTTPMethod {
        switch self {
        case .get: return .get
        case .post: return .post
        }
    }
    var path: String {
        switch self {
        case .get: return "get"
        case .post: return "post"
        case .put: return "put"  
        }
    }
    func asURLRequest() throws -> URLRequest {
        let url = baseURL.appendingPathComponent(path)
        var request = URLRequest(url: url)
        request.method = method
        switch self {
        case let .get(parameters):
            request = try URLEncodedFormParameterEncoder().encode(parameters, into: request)
        case let .post(parameters):
            request = try JSONParameterEncoder().encode(parameters, into: request)
        }
        return request
    }
}
```

### ParameterEncoding & ParameterEncoder

共同点：

+ 在`Session中`创建`Request`时使用
+ 把参数编码进`URLRequest`中使用
+ 决定参数的编码位置（url query string、body表单、bodyjson）

不同点

+ 初始化参数不同。ParameterEncoding只能编码字典数据， ParameterEncoder用来编码任意实现Encodable协议的数据类型。
+ ParameterEncoding只有在创建DataRequest跟DownloadRequest时使用，DataStreamRequest无法使用；而ParameterEncoder这三个Request子类都能用来初始化。

#### ParameterEncoding协议

`ParameterEncoding`协议跟`ParameterEncoder`协议的不同之处在于参数不再需要遵循 `Encodable` 协议，改为直接使用 `typealias Parameters = [String: Any]` ， 即 `key` 类型指定为 `String` 的 `Dictionary`。

```swift
/// 用于' URLRequest '的参数字典
public typealias Parameters = [String: Any]
/// 用于定义如何将一组参数应用于' URLRequest '的类型
public protocol ParameterEncoding {
    /// 使用URLRequestConvertible创建URLRequest, 然后把字典参数编码进URLRequest中, 可以抛出异常, 抛出异常时会返回AFError.parameterEncodingFailed错误
    func encode(_ urlRequest: URLRequestConvertible, with parameters: Parameters?) throws -> URLRequest
}
```

`URLEncoding`默认实现，用来编码`url query string`。

```swift
public struct URLEncoding: ParameterEncoding {
  
		 /// 定义参数被编码到url query中还是body中
    public enum Destination {
    	... 
    }
  	/// 决定如何编码Array
    public enum ArrayEncoding {
      ...
    }
  	///决定如何编码Bool
    public enum BoolEncoding {
      ...
    }
  
  	// MARK: 快速初始化的三个静态计算属性
    /// 默认使用method决定编码位置, 数组使用带括号, bool使用数字
    public static var `default`: URLEncoding { URLEncoding() }
    /// url query 编码, 数组使用带括号, bool使用数字
    public static var queryString: URLEncoding { URLEncoding(destination: .queryString) }
    /// form 表单编码到body, 数组使用带括号, bool使用数字
    public static var httpBody: URLEncoding { URLEncoding(destination: .httpBody) }
  	
  	// MARK: 属性与初始化
    /// 参数编码位置
    public let destination: Destination
    /// 数组编码格式
    public let arrayEncoding: ArrayEncoding
    /// Bool编码格式
    public let boolEncoding: BoolEncoding
    public init(destination: Destination = .methodDependent,
                arrayEncoding: ArrayEncoding = .brackets,
                boolEncoding: BoolEncoding = .numeric) {
        self.destination = destination
        self.arrayEncoding = arrayEncoding
        self.boolEncoding = boolEncoding
    }
  
  	// MARK: 实现协议的编码方法
  	public func encode(_ urlRequest: URLRequestConvertible, with parameters: Parameters?) throws -> URLRequest {
      ...
    }
  	/// 对key-value对进行编码, value主要处理字典,数组,nsnumber类型的bool,bool以及其他值
    public func queryComponents(fromKey key: String, value: Any) -> [(String, String)] {
      ...
    }
  	/// url转义, 转成百分号格式的
    /// 会忽略 :#[]@!$&'()*+,;=
    public func escape(_ string: String) -> String {
        string.addingPercentEncoding(withAllowedCharacters: .afURLQueryAllowed) ?? string
    }
    /// 把参数字典转成query string
    private func query(_ parameters: [String: Any]) -> String {
      ...
    }
}
```

`JSONEncoding`默认实现，用来把参数编码成json丢入body中。使用`JSONSerialization`来把参数字典编码为json, 一定会被编码到body中, 并且会设置`Content-Type为application/json`。

```swift
public struct JSONEncoding: ParameterEncoding {
    // MARK: 用来快速初始化的静态计算变量
  	/// 默认类型, 压缩json格式
    public static var `default`: JSONEncoding { JSONEncoding() }
  	//标准json格式
    public static var prettyPrinted: JSONEncoding { JSONEncoding(options: .prettyPrinted) }

    // MARK: 属性与初始化
    //保存JSONSerialization.WritingOptions
    public let options: JSONSerialization.WritingOptions

    public init(options: JSONSerialization.WritingOptions = []) {
        self.options = options
    }

    // MARK: 实现协议的编码方法
  	public func encode(_ urlRequest: URLRequestConvertible, with parameters: Parameters?) throws -> URLRequest {
      ...
    }
  	/// 把json对象编码进body中, 其实上面的编码方法可以直接掉这个方法, 两个方法实现一样
    public func encode(_ urlRequest: URLRequestConvertible, withJSONObject jsonObject: Any? = nil) throws -> URLRequest {
      ...
    }
}
```

#### ParameterEncoder协议

`ParameterEncoder`协议，一个参数编码器，用来编码支持`Encodable` 协议类型到`URLRequest`中，只需要实现一个方法用于处理 `parameters` 并生成对应的 `URLRequest` 。

```swift
public protocol ParameterEncoder {
    func encode<Parameters: Encodable>(_ parameters: Parameters?, into request: URLRequest) throws -> URLRequest
}
```

`JSONParameterEncoder`用于`JSON` 的编码，如果`URLRequest` 的请求头没有设置`Content-Type` ，它就会设置为 `application/json`。`JSONParameterEncoder`只支持对 `URLRequest` 的 `httpBody` 属性进行设置。

通过`JSONParameterEncoder`编码json数据:

```swift
/// 实现协议的编码方法:
open func encode<Parameters: Encodable>(_ parameters: Parameters?, into request: URLRequest) throws -> URLRequest {   
    guard let parameters = parameters else { return request }
    var request = request
    do {
        /// 把参数编码成json data
        let data = try encoder.encode(parameters)
        /// 设置httpBody数据
        request.httpBody = data
        /// 设置Content-Type
        if request.headers["Content-Type"] == nil {
            request.headers.update(.contentType("application/json"))
        }
    } catch {
        /// 解析json异常就抛出错误
        throw AFError.parameterEncodingFailed(reason: .jsonEncodingFailed(error: error))
    }
    return request
}
```

`URLEncodedFormParameterEncoder`用于生成`URL`编码方式的字符串，与`JSONParameterEncoder` 不同，可以追加到 URL 后面，也可以设置到 body 中，取决于`Destination` 。

```swift
enum Destination {
  	 /// 如果是 .get， .head 和 .delete 方法则追加到 URL 链接后面，否则设置为 httpBody
		 case methodDependent 
  	 /// 全都追加到 URL 链接后面
  	 case methodDependent 
  	 // 全都设置为 httpBody 
		 case httpBody 
}
```

`URLEncodedFormParameterEncoder` 的 `encoder` 属性是 `URLEncodedFormEncoder` 类型，用于编码时各种类型的转换。

```swift
open func encode<Parameters: Encodable>(_ parameters: Parameters?,
                                        into request: URLRequest) throws -> URLRequest {
    guard let parameters = parameters else { return request }

    var request = request
	  /// 判断request是否有url ，如果没有则报parameterEncoderFailed
    guard let url = request.url else {
        throw AFError.parameterEncoderFailed(reason: .missingRequiredComponent(.url))
    }
    /// 判断request是否有指定method ，如果没有则报parameterEncoderFailed
    guard let method = request.method else {
        let rawValue = request.method?.rawValue ?? "nil"
        throw AFError.parameterEncoderFailed(reason: .missingRequiredComponent(.httpMethod(rawValue: rawValue)))
    }
    /// 判断参数编码后是否追加到url中
    if destination.encodesParametersInURL(for: method),
        var components = URLComponents(url: url, resolvingAgainstBaseURL: false) {
        /// mapError 用于转换 Result 为 failure 时的 error ，get() 可以获取 success value
        let query: String = try Result<String, Error> { try encoder.encode(parameters) }
            .mapError { AFError.parameterEncoderFailed(reason: .encoderFailed(error: $0)) }.get()
        let newQueryString = [components.percentEncodedQuery, query].compactMap { $0 }.joinedWithAmpersands()
        components.percentEncodedQuery = newQueryString.isEmpty ? nil : newQueryString
        guard let newURL = components.url else {
            throw AFError.parameterEncoderFailed(reason: .missingRequiredComponent(.url))
        }
        request.url = newURL
    } else {
        /// 参数编码后设置为 httpBody
        if request.headers["Content-Type"] == nil {
            request.headers.update(.contentType("application/x-www-form-urlencoded; charset=utf-8"))
        }
        request.httpBody = try Result<Data, Error> { try encoder.encode(parameters) }
            .mapError { AFError.parameterEncoderFailed(reason: .encoderFailed(error: $0)) }.get()
    }

    return request
}
```

### Notifications

`Notifications.swift` 的结构由`Request`、`Notification`、`AlamofireNotifications`三个部分组成。

1.`Request` 的扩展，定义了请求相关的通知，通过 `static let` 定义相关通知，方便调用：

```swift
public extension Request {
    static let didResumeNotification = Notification.Name(rawValue: “org.alamofire.notification.name.request.didResume”)
}
/// 调用
Request.didResumeNotification
```

2.`Notification` 和 `NotificationCenter` 的扩展，方便与 `Request` 进行交互：

```swift
extension Notification {
    /// 把userInfo 的 Request 通过 String.requestKey 封装起来，方便获取
    public var request: Request? {
        return userInfo?[String.requestKey] as? Request
    }

    /// 通过 Request 和 NotificationName 生成 Notification ，不需要每次都手动设置 userInfo
    init(name: Notification.Name, request: Request) {
        self.init(name: name, object: nil, userInfo: [String.requestKey: request])
    }
}
extension NotificationCenter {
    /// Convenience function for posting notifications with `Request` payloads.
    func postNotification(named name: Notification.Name, with request: Request) {
        let notification = Notification(name: name, request: request)
        post(notification)
    }
}
```

3.定义了一个 `AlamofireNotifications` 类，遵循 `EventMonitor`协议，这个类的作用下面会说到，通过 `AlamofireNotifications` 我们可以在需要的地方添加 `Request` 的相关通知，灵活地实现对应的方法，不需要统一配置，而发送通知的时机也嵌入到 `EventMonitor` 的逻辑中，不需要额外处理：

```swift
public final class AlamofireNotifications: EventMonitor {
    public func requestDidResume(_ request: Request) {
        NotificationCenter.default.postNotification(named: Request.didResumeNotification, with: request)
    }
    public func requestDidSuspend(_ request: Request) {
        NotificationCenter.default.postNotification(named: Request.didSuspendNotification, with: request)
    }
    public func requestDidCancel(_ request: Request) {
        NotificationCenter.default.postNotification(named: Request.didCancelNotification, with: request)
    }
    public func requestDidFinish(_ request: Request) {
        NotificationCenter.default.postNotification(named: Request.didFinishNotification, with: request)
    }
    public func request(_ request: Request, didResumeTask task: URLSessionTask) {
        NotificationCenter.default.postNotification(named: Request.didResumeTaskNotification, with: request)
    }
    public func request(_ request: Request, didSuspendTask task: URLSessionTask) {
        NotificationCenter.default.postNotification(named: Request.didSuspendTaskNotification, with: request)
    }
    public func request(_ request: Request, didCancelTask task: URLSessionTask) {
        NotificationCenter.default.postNotification(named: Request.didCancelTaskNotification, with: request)
    }
    public func request(_ request: Request, didCompleteTask task: URLSessionTask, with error: AFError?) {
        NotificationCenter.default.postNotification(named: Request.didCompleteTaskNotification, with: request)
    }
}

```

### Protected

`Protected<T>` 支持范型，所以我们可以通过 `Protected<T>` 为各个类，结构体提供线程安全的封装。为了实现线程安全，一般都需要一把锁，`Protected.swift`里定义了一个`private protocol Lock` ，Protected.swift中对`defer` 的使用也非常巧妙

```swift
private protocol Lock {
    func lock()
    func unlock()
}
extension Lock {
    // 执行有返回值的closure
    func around<T>(_ closure: () -> T) -> T {
        lock(); defer { unlock() }
        return closure()
    }
    /// 执行无返回值的closure
    func around(_ closure: () -> Void) {
        lock(); defer { unlock() }
        closure()
    }
}

```

遵循`Lock`协议需要实现两个方法`lock()`和`unlock()` ，而在`Lock` 的`extension`里面，提供了两个结合 `closure` 实现加锁的方法，这两个方法可以满足大部分需要加锁的操作需求。`Alamofire`需要支持Linux ，而`Linux`无法使用 `os_unfair_lock`，只能使用`pthread_mutex_t` 。

```swift
#if os(Linux)
/// 5.1 新增的 MutexLock ，是 pthread_mutex_t 的 wrapper ，为了支持 Linux 平台。
final class MutexLock: Lock {
		 private var mutex: UnsafeMutablePointer<pthread_mutex_t>
  	 ...
}
#endif
#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
/// 原有的 os_unfair_lock ，苹果平台使用。
final class UnfairLock: Lock {
    private let unfairLock: os_unfair_lock_t
  	...
}
#endif
```

在定义了 `Lock` 类之后，`Protected<T>` 就可以通过 `Lock` 来实现线程安全，这里使用了 `@propertyWrapper` 来进行声明。

```swift
@propertyWrapper
final class Protected<T> {
    #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
    private let lock = UnfairLock()
    #elseif os(Linux)
    private let lock = MutexLock()
    #endif
    private var value: T
    init(_ value: T) {
        self.value = value
    }
    /// 访问 wrappedValue 时必须要加锁。
    var wrappedValue: T {
        get { lock.around { value } }
        set { lock.around { value = newValue } }
    }
    var projectedValue: Protected<T> { self }
    init(wrappedValue: T) {
        value = wrappedValue
    }
    /// 同步获取值或者进行转换
    func read<U>(_ closure: (T) -> U) -> U {
        lock.around { closure(self.value) }
    }
    /// 同步修改值，同时可以返回修改后的值。
    @discardableResult
    func write<U>(_ closure: (inout T) -> U) -> U {
        lock.around { closure(&self.value) }
    }
  	/// 支持 @dynamicMemberLookup 后实现的方法，使得 Protected 声明的属性可以通过点语法来进行 keyPath 读写
    subscript<Property>(dynamicMember keyPath: WritableKeyPath<T, Property>) -> Property {
        get { lock.around { value[keyPath: keyPath] } }
        set { lock.around { value[keyPath: keyPath] = newValue } }
    }
}
```

为了方便使用， `Protected`还添加了几个扩展。 这部分是为`T 实现 `RangeReplaceableCollection`协议后支持的方法，都是集合的 `append` 方法：

```swift
extension Protected where T: RangeReplaceableCollection {
    func append(_ newElement: T.Element) {
        write { (ward: inout T) in
            ward.append(newElement)
        }
    }
    func append<S: Sequence>(contentsOf newElements: S) where S.Element == T.Element {
        write { (ward: inout T) in
            ward.append(contentsOf: newElements)
        }
    }
    func append<C: Collection>(contentsOf newElements: C) where C.Element == T.Element {
        write { (ward: inout T) in
            ward.append(contentsOf: newElements)
        }
    }
}
```

`T`为`Data?` 类型时支持`append` :

```swift
extension Protected where T == Data? {
    func append(_ data: Data) {
        write { (ward: inout T) in
            ward?.append(data)
        }
    }
}
```

`T`为`Request.MutableState`时提供一些编辑方法，用于状态转换：

```swift
extension Protected where T == Request.MutableState {
    func attemptToTransitionTo(_ state: Request.State) -> Bool {
        lock.around {
            guard value.state.canTransitionTo(state) else { return false }
            value.state = state
            return true
        }
    }
    func withState(perform: (Request.State) -> Void) {
        lock.around { perform(value.state) }
    }
}
```

### EventMonitor

`EventMonitor`协议可以用来获取`Request`各种方法和状态的相关回调，一种最常见的用法就是打印日志：

```swift
final class Logger: EventMonitor {
    let queue = DispatchQueue(label: ...)
    func requestDidResume(_ request: Request) {
        print("Resuming: \(request)")
    }
    func request<Value>(_ request: DataRequest, didParseResponse response: DataResponse<Value, AFError>) {
        debugPrint("Finished: \(response)")
    }
}
let logger = Logger()
let session = Session(eventMonitors: [logger])
```

Alamofire里面为了让调用方可以更灵活地使用`EventMonitor` ，添加了一个`extension` ，里面实现了`EventMonitor` 的全部方法和属性，这样调用方就不需要实现`EventMonitor` 的全部方法，可以根据自己需要添加对应的方法即可。 Alamofire 实现了一个 `CompositeEventMonitor` 类，用于组合各个 `EventMonitors`。

```swift
public final class CompositeEventMonitor: EventMonitor {
    public let queue = DispatchQueue(label: "org.alamofire.compositeEventMonitor", qos: .utility)

    let monitors: [EventMonitor]

    init(monitors: [EventMonitor]) {
        self.monitors = monitors
    }
		...
}
```

### Result

`Alamofire`使用了`typealias` 定义了一个 `AFResult` ，其中`Failure` 类型固定为 `AFError`：

```swift
public typealias AFResult<Success> = Result<Success, AFError>
```

`Success`与`AFError`

```swift
/// 判断是否为 .success 和 .failure
var isSuccess: Bool {
    guard case .success = self else { return false }
    return true
}
var isFailure: Bool {
    !isSuccess
}
/// 获取.success和.failure对应的值
var success: Success? {
    guard case let .success(value) = self else { return nil }
    return value
}
var failure: Failure? {
    guard case let .failure(error) = self else { return nil }
    return error
}
```

初始化`Result`

```swift
init(value: Success, error: Failure?) {
    if let error = error {
        self = .failure(error)
    } else {
        self = .success(value)
    }
}
```

`tryMap`对`Success`进行转换，在转换过程中可以抛出`Error` ：

```swift
func tryMap<NewSuccess>(_ transform: (Success) throws -> NewSuccess) -> Result<NewSuccess, Error> {
    switch self {
    case let .success(value):
        do {
            return try .success(transform(value))
        } catch {
            return .failure(error)
        }
    case let .failure(error):
        return .failure(error)
    }
}
```

`tryMapError`对`Failure` 进行转换，在转换过程中也可以抛出`Error` ：

```swift
func tryMapError<NewFailure: Error>(_ transform: (Failure) throws -> NewFailure) -> Result<Success, Error> {
    switch self {
    case let .failure(error):
        do {
            return try .failure(transform(error))
        } catch {
            return .failure(error)
        }
    case let .success(value):
        return .success(value)
    }
}
```

`Result`中`tryMap`用法如下:

```swift
let possibleData: Result<Data, Error> = .success(Data(...))
let possibleObject = possibleData.tryMap {
    try JSONSerialization.jsonObject(with: $0)
}
```

`Result`中`tryMapError`用法如下:

```swift
let possibleData: Result<Data, Error> = .success(Data(...))
let possibleObject = possibleData.tryMapError {
    try someFailableFunction(taking: $0)
}
```

### Response

`Alamofire`对请求结果进行封装，根据类型不同分为`DataResponse` 、`DownloadResponse`、`DataRequest` 和`UploadRequest` ，请求返回的结果是 `DataResponse<Success, Failure: Error>` 。

```swift
public struct DataResponse<Success, Failure: Error> {
    /// 对应的 URLRequest
    public let request: URLRequest?
    /// 服务器返回的 HTTPURLResponse
    public let response: HTTPURLResponse?
    /// 服务器返回的 Data
    public let data: Data?
    /// 响应对应的 URLSessionTaskMetrics
    public let metrics: URLSessionTaskMetrics?
    /// 序列化所消耗的时间
    public let serializationDuration: TimeInterval
    /// 序列化的结果
    public let result: Result<Success, Failure>
    public var value: Success? { result.success }
    public var error: Failure? { result.failure }
    public init(request: URLRequest?,
                response: HTTPURLResponse?,
                data: Data?,
                metrics: URLSessionTaskMetrics?,
                serializationDuration: TimeInterval,
                result: Result<Success, Failure>) {
        self.request = request
        self.response = response
        self.data = data
        self.metrics = metrics
        self.serializationDuration = serializationDuration
        self.result = result
    }
}
```

`DownloadRequest` 对应的则是 `DownloadResponse<Success, Failure: Error>` ，大部分属性跟 `DataResponse` 一致，新增以下两个属性：

```swift
/// 用于存储响应的数据的文件 URL
public let fileURL: URL?
/// 取消请求时所接收到的数据
public let resumeData: Data?
```

### AFError

请求错误`AFError`是枚举`enum`类型，继承`Error`协议，由于网络请求错误类型比较多，所以`AFError`定义两层，第一层是`AFError`，而另一层则是基于`AFError`的枚举`enum`类型，`enum`类型是细化`AFError`网络请求错误类型。

```swift
MultipartEncodingFailureReason 			// 表单转码错误
ParameterEncodingFailureReason 			// 参数转码错误
ParameterEncoderFailureReason 			// 参数转码器错误
RequiredComponent 									// 缺少必要的组件
ResponseValidationFailureReason 		// 响应数据验证错误
ResponseSerializationFailureReason 	// 响应数据序列化错误
ServerTrustFailureReason 						// 服务器验证错误
URLRequestValidationFailureReason 	// 请求验证错误
```

还有`Alamofire`提供了不少扩展方法给`Error`和`AFError`使用，以求在使用上更便捷。如`Error`可以转化为 `AFError`: 

```swift
extension Error {
    /// Returns the instance cast as an `AFError`.
    public var asAFError: AFError? {
        self as? AFError
    }
    public func asAFError(orFailWith message: @autoclosure () -> String, file: StaticString = #file, line: UInt = #line) -> AFError {
        guard let afError = self as? AFError else {
            fatalError(message(), file: file, line: line)
        }
        return afError
    }
    func asAFError(or defaultAFError: @autoclosure () -> AFError) -> AFError {
        self as? AFError ?? defaultAFError()
    }
}
```

## Alamofire请求流程

所有`Alamofire.swift`的请求方法都会调用`Session`里面对应的方法，而`Session`负责创建和管理`Alamofire`的 `Request`类。同时也提供一些公共的方法给所有`Request`使用，如请求队列、信任管理、重定向处理和响应缓存处理等。

创建一个`Request`的子类后， `Alamofire`会进行一系列的操作来完成这个请求，请求流程如下:

+ 初始化一些参数，比如 HTTP 方法、 HTTP 头和参数等会被封装进内部的`URLRequestConvertible`值中，用于初始化`Request`。
+ 调用封装好的`URLRequestConvertible`的`asURLRequest()`方法来创建第一个`URLRequest`，`URLRequest` 会存储到`Request`的`mutableState.requests`属性中。
+ 如果`Session`或者`Request`有提供`RequestAdapters`或者`RequestInterceptors` ，则会对之前生成的 `URLRequest`进行调整，同时也会存储到`Request`的`mutableState.requests`属性中。
+ `Session`调用`Request`的方法来生成对应的`URLSessionTask` ，不同的`Request`子类 会生成不同的 `URLSessionTask`。
+ 当`URLSessionTask`完成任务，且已经收集到`URLSessionTaskMetrics` ，`Request`就会执行自己的`Validators `来验证请求结果是否正确。
+ 通过验证后，`Request`就会执行`mutableState.responseSerializers`来处理请求结果。

上面这些步骤中，每个步骤都有可能产生错误或者接收到网络返回的错误结果，这些错误会传递给对应的`Request` 。在处理结果时会判断是否需要重试。 当`Error`传递给`Request`后，`Request`会调用对应的`RequestRetriers`来判断是否需要进行重试，如果需要进行重试，则再走一次上面的流程，`Alamofire`请求流程图如下:

![](/assets/images/swift-alamofire-02.png)

### DataRequest

`Alamofire` 为发起`DataRequest` 提供了三个请求接口：

定义一个`struct RequestConvertible` 来支持`URLRequestConvertible`协议，负责处理参数的编码和生成对应的`URLRequest`。

```swift
public typealias RequestModifier = (inout URLRequest) throws -> Void
struct RequestConvertible: URLRequestConvertible {
    let url: URLConvertible
    let method: HTTPMethod
    let parameters: Parameters?
    let encoding: ParameterEncoding
    let headers: HTTPHeaders?
    let requestModifier: RequestModifier?
    func asURLRequest() throws -> URLRequest {
        var request = try URLRequest(url: url, method: method, headers: headers)
        try requestModifier?(&request)
        return try encoding.encode(request, with: parameters)
    }
}
```

从传递的组件和`RequestInterceptor`创建的`URLRequest`中创建一个`DataRequest`。

```swift
/// parameters为Parameters ，即 [String: Any]
open func request(_ convertible: URLConvertible,
                      method: HTTPMethod = .get,
                      parameters: Parameters? = nil,
                      encoding: ParameterEncoding = URLEncoding.default,
                      headers: HTTPHeaders? = nil,
                      interceptor: RequestInterceptor? = nil,
                      requestModifier: RequestModifier? = nil) -> DataRequest {
    let convertible = RequestConvertible(url: convertible,
                                             method: method,
                                             parameters: parameters,
                                             encoding: encoding,
                                             headers: headers,
                                             requestModifier: requestModifier)
    return request(convertible, interceptor: interceptor)
}
```

定义一个`struct RequestConvertible` ，支持`URLRequestConvertible`协议，负责处理参数的编码和生成对应的`URLRequest` ：

```swift
public typealias RequestModifier = (inout URLRequest) throws -> Void
struct RequestConvertible: URLRequestConvertible {
    let url: URLConvertible
    let method: HTTPMethod
    let parameters: Parameters?
    let encoding: ParameterEncoding
    let headers: HTTPHeaders?
    let requestModifier: RequestModifier?
    func asURLRequest() throws -> URLRequest {
        var request = try URLRequest(url: url, method: method, headers: headers)
        try requestModifier?(&request)
        return try encoding.encode(request, with: parameters)
    }
}
```

通过使用传递的组件、可编码参数和`RequestInterceptor`创建的' URLRequest '中创建一个' DataRequest '，由于参数编码方式不同，所以需要定义一个`struct RequestEncodableConvertible<Parameters: Encodable>` ，来处理参数为`Encodable`时的编码和生成 `URLRequest`。

```swift
/// parameters支持Encodable
open func request<Parameters: Encodable>(_ convertible: URLConvertible,
                                             method: HTTPMethod = .get,
                                             parameters: Parameters? = nil,
                                             encoder: ParameterEncoder = URLEncodedFormParameterEncoder.default,
                                             headers: HTTPHeaders? = nil,
                                             interceptor: RequestInterceptor? = nil,
                                             requestModifier: RequestModifier? = nil) -> DataRequest {
    let convertible = RequestEncodableConvertible(url: convertible,
                                                      method: method,
                                                      parameters: parameters,
                                                      encoder: encoder,
                                                      headers: headers,
                                                      requestModifier: requestModifier)
    return request(convertible, interceptor: interceptor)
}
```

上面两个接口经过处理生成对应的 `URLRequestConvertible` 后会调用这个接口来生成 `DataRequest` ，在使用的时候也可以自己生成 `URLRequestConvertible` ，直接调用这个接口。

```swift
/// 通过URLRequestConvertible的值和RequestInterceptor适配器创建DataRequest
open func request(_ convertible: URLRequestConvertible, interceptor: RequestInterceptor? = nil) -> DataRequest {
    let request = DataRequest(convertible: convertible,
                                  underlyingQueue: rootQueue,
                                  serializationQueue: serializationQueue,
                                  eventMonitor: eventMonitor,
                                  interceptor: interceptor,
                                  delegate: self)
    perform(request)
    return request
}
```

这样可以统一 `DataRequest` 的处理流程，虽然可以通过不同的参数来生成 `DataRequest` ，但是在 `Session` 内部的处理时，会通过 `URLRequestConvertible` 进行转换，收敛到同一个接口中。

### Perform

在这里的`perform`只是用来进行进行请求前的准备工作，并没有发起真正的网络请求。

```swift
func perform(_ request: Request) {
    rootQueue.async {
        /// 判断 request 是否有取消
        guard !request.isCancelled else { return }
        /// 将request添加到activeRequests中
        self.activeRequests.insert(request)
        self.requestQueue.async {
            switch request {
            /// UploadRequest must come before DataRequest due to subtype relationship.  
            case let r as UploadRequest: self.performUploadRequest(r) 
            case let r as DataRequest: self.performDataRequest(r)
            case let r as DownloadRequest: self.performDownloadRequest(r)
            case let r as DataStreamRequest: self.performDataStreamRequest(r)
            default: fatalError("Attempted to perform unsupported Request subclass: \(type(of: request))")
            }
        }
    }
}
```

期间调用`performSetupOperations`方法进行一些请求前的准备工作。 在处理过程中会判断是否有设置`adapter` ，如果有设置 `adapter` ，则调用`adapter`对请求做一次适配。

```swift
func performSetupOperations(for request: Request, convertible: URLRequestConvertible) {
   	...
}
```

`didCreateURLRequest` 则负责创建对应的 `URLSessionTask` ，在 `requestTaskMap` 中添加对应的记录。

```swift
func didCreateURLRequest(_ urlRequest: URLRequest, for request: Request) {
    dispatchPrecondition(condition: .onQueue(rootQueue))
    request.didCreateURLRequest(urlRequest)
    guard !request.isCancelled else { return }
    let task = request.task(for: urlRequest, using: session)
    requestTaskMap[request] = task
    request.didCreateTask(task)
    // 根据 request 的 state 对 task 进行操作
    updateStatesForTask(task, request: request)
}
```

```swift
func updateStatesForTask(_ task: URLSessionTask, request: Request) {
    dispatchPrecondition(condition: .onQueue(rootQueue))
    request.withState { state in
        switch state {
        case .initialized, .finished:
            break
        case .resumed:
            task.resume()
            rootQueue.async { request.didResumeTask(task) }
        case .suspended:
            task.suspend()
            rootQueue.async { request.didSuspendTask(task) }
        case .cancelled:
            task.resume()
            task.cancel()
            rootQueue.async { request.didCancelTask(task) }
        }
    }
}
```

### RedirectHandler重定向

`Alamofire`提供了一个重定向协议`RedirectHandler` ，提供了以下方法，可以生成新的`URLRequest` ，也可以直接调用 `completion(nil)` 来拒绝重定向。

```swift
public protocol RedirectHandler {
    func task(_ task: URLSessionTask,
              willBeRedirectedTo request: URLRequest,
              for response: HTTPURLResponse,
              completion: @escaping (URLRequest?) -> Void)
}
let redirector = Redirector(behavior: .follow)
Alamofire.request(...)
    .redirect(using: redirector)
    .responseDecodable(of: SomeType.self) { response in 
        debugPrint(response)
    }
```

并且`Redirector`为`Alamofire`提供了的一个重定向默认实现，在重定向时根据`behavior`来判断如何进行重定向的相关操作。

```swift
public enum Behavior {
    case follow
    case doNotFollow
    case modify((URLSessionTask, URLRequest, HTTPURLResponse) -> URLRequest?)
}
extension Redirector: RedirectHandler {
    public func task(_ task: URLSessionTask,
                     willBeRedirectedTo request: URLRequest,
                     for response: HTTPURLResponse,
                     completion: @escaping (URLRequest?) -> Void) {
        switch behavior {
        case .follow:
            completion(request)
        case .doNotFollow:
            completion(nil)
        case let .modify(closure):
            let request = closure(task, request, response)
            completion(request)
        }
    }
}
```

在`SessionDelegate` 中也会调用对应的 `redirectHandler` 进行重定向。

```swift
open func urlSession(_ session: URLSession,
                     task: URLSessionTask,
                     willPerformHTTPRedirection response: HTTPURLResponse,
                     newRequest request: URLRequest,
                     completionHandler: @escaping (URLRequest?) -> Void) {
    eventMonitor?.urlSession(session, task: task, willPerformHTTPRedirection: response, newRequest: request)
    // 获取对应的 redirectHandler ，如果 request 的 redirectHandler 为 nil ，就尝试获取 session 的 redirectHandler
    if let redirectHandler = stateProvider?.request(for: task)?.redirectHandler ?? stateProvider?.redirectHandler {
        redirectHandler.task(task, willBeRedirectedTo: request, for: response, completion: completionHandler)
    } else {
        completionHandler(request)
    }
}
```

### 自定义缓存

Alamofire 提供了一个自定义协议`CachedResponseHandler` ，用于判断是否需要缓存当前的`HTTP`响应结果，可以生成新的 `CachedURLResponse`，也可以调用`completion(nil)`来拒绝进行缓存。

```swift
public protocol CachedResponseHandler {
    func dataTask(_ task: URLSessionDataTask,
                  willCacheResponse response: CachedURLResponse,
                  completion: @escaping (CachedURLResponse?) -> Void)
}
```

`ResponseCacher`是`Alamofire`提供的遵循`CachedResponseHandler`协议的一个类，也是通过`Behavior`来判断如何实现缓存.

```swift
open func urlSession(_ session: URLSession,
                     dataTask: URLSessionDataTask,
                     willCacheResponse proposedResponse: CachedURLResponse,
                     completionHandler: @escaping (CachedURLResponse?) -> Void) {
    eventMonitor?.urlSession(session, dataTask: dataTask, willCacheResponse: proposedResponse)
    // 获取对应的 cachedResponseHandler ，如果 request 的 cachedResponseHandler 为 nil ，就尝试获取 session 的 cachedResponseHandler
    if let handler = stateProvider?.request(for: dataTask)?.cachedResponseHandler ?? stateProvider?.cachedResponseHandler {
        handler.dataTask(dataTask, willCacheResponse: proposedResponse, completion: completionHandler)
    } else {
        completionHandler(proposedResponse)
    }
}
```

### Request处理请求结果

![](/assets/images/swift-alamofire-03.png)

 `ResponseSerializer`完成处理后都会调用`responseSerializerDidComplete(completion: @escaping () -> Void)` ，获取下一个 `nextResponseSerializer()` 。

`DataRequest`对结果不做任何序列化操作的方法， 对应的方法实现。

```swift
func response(queue: DispatchQueue = .main, completionHandler: @escaping (AFDataResponse<Data?>) -> Void) -> Self {
    appendResponseSerializer {
        let result = AFResult<Data?>(value: self.data, error: self.error)
		  	/// 完成序列化操作后，转换到underlyingQueue中进行处理，因为在进行序列化操作时是在serializationQueue进行处理
        self.underlyingQueue.async {
        /// 生成对应的DataResponse，因为不做任何序列化操作，所以serializationDuration直接设置0
			  let response = DataResponse(request: self.request,
                                        response: self.response,
                                        data: self.data,
                                        metrics: self.metrics,
                                        serializationDuration: 0,
                                        result: result)
            self.eventMonitor?.request(self, didParseResponse: response)
			  		/// 调用responseSerializerDidComplete添加completionHandler
            self.responseSerializerDidComplete { queue.async { completionHandler(response) } }
        }
    }
    return self
}
```

添加 `ResponseSerializer`.

```swift
func appendResponseSerializer(_ closure: @escaping () -> Void) {
    $mutableState.write { mutableState in
        mutableState.responseSerializers.append(closure)
        /// 把状态由.finished为.resumed, 原因是如果ResponseSerializer序列化失败，会重新发送请求。
        /// 而重新发送请求时会调用 updateStatesForTask 方法，只有request的状态为resumed时才会调用task.resume() 
        /// 所以这里要设置为resumed，使得可以调用task.resume()，重新发送请求
        if mutableState.state == .finished {
            mutableState.state = .resumed
        }
        /// 判断是否已经处理完ResponseSerializer，如果已经处理完，则直接开始处理新增的ResponseSerializer
        if mutableState.responseSerializerProcessingFinished {
          	/// processNextResponseSerializer
            underlyingQueue.async { self.processNextResponseSerializer() }
        }
        /// 是否需要直接开始进行网络请求
        if mutableState.state.canTransitionTo(.resumed) {
            underlyingQueue.async { if self.delegate?.startImmediately == true { self.resume() } }
        }
    }
}
```

对下一个`ResponseSerializer`进行处理，如果所有`ResponseSerializer`都处理完毕则调用所有的`completions`.

```swift
func processNextResponseSerializer() {
    guard let responseSerializer = nextResponseSerializer() else {
        var completions: [() -> Void] = []
        $mutableState.write { mutableState in
            completions = mutableState.responseSerializerCompletions
				   	/// 如果已经完成所有序列化操作，优先移除所有的ResponseSerializers和ResponseSerializerCompletions ，
            /// 执行所有ResponseSerializerCompletions
            mutableState.responseSerializers.removeAll()
            mutableState.responseSerializerCompletions.removeAll()
            if mutableState.state.canTransitionTo(.finished) {
                mutableState.state = .finished
            }
            mutableState.responseSerializerProcessingFinished = true
            mutableState.isFinishing = false
        }
        /// 执行所有ResponseSerializerCompletions
        completions.forEach { $0() }
        cleanup()
        return
    }
	 /// 如果还有序列化操作未执行，就先执行
   serializationQueue.async { responseSerializer() }
}
```

获取下一个序列化操作，因为只有完成了序列化操作后才会添加对应的 `ResponseSerializerCompletion` ，所以通过 `responseSerializers`和`responseSerializerCompletions` 的`count`来比较即可知道是否还有序列化操作未处理.

```swift
func nextResponseSerializer() -> (() -> Void)? {
    var responseSerializer: (() -> Void)?
    $mutableState.write { mutableState in
        let responseSerializerIndex = mutableState.responseSerializerCompletions.count
        if responseSerializerIndex < mutableState.responseSerializers.count {
            responseSerializer = mutableState.responseSerializers[responseSerializerIndex]
        }
    }
    return responseSerializer
}
```

### Serializer序列化操作

`Alamofire`提供了`DataResponseSerializerProtocol`和`DownloadResponseSerializerProtocol`协议，用于添加一些自定义的序列化操作.

```swift
/// 定义了一个关联类型SerializedObject和一个用于将原始数据转换为SerializedObject的方法
public protocol DataResponseSerializerProtocol {
    associatedtype SerializedObject
    func serialize(request: URLRequest?, response: HTTPURLResponse?, data: Data?, error: Error?) throws -> SerializedObject
}
public protocol DownloadResponseSerializerProtocol {
    associatedtype SerializedObject
    func serializeDownload(request: URLRequest?, response: HTTPURLResponse?, fileURL: URL?, error: Error?) throws -> SerializedObject
}
```

`DataRequest`提供了以下方法给我们添加自定义的序列化操作:

```swift
public func response<Serializer: DataResponseSerializerProtocol>(queue: DispatchQueue = .main,
                                                                 responseSerializer: Serializer,
                                                                 completionHandler: @escaping (AFDataResponse<Serializer.SerializedObject>) -> Void)
    -> Self {
      ...
    }
```

## NetworkReachabilityManager

`NetworkReachabilityManager`是 Alamofire 提供的用于检测网络状态的工具类。跟[Reachability.swift](https://github.com/ashleymills/Reachability.swift)功能类似，知识`NetworkReachabilityManager`只用获取监测网络状态，在设备的网络状态改变时收到通知，不要以此来判断是否可以连接某个 host 或者地址。

### NetworkReachabilityStatus

`NetworkReachabilityManager定义`NetworkReachabilityStatus`的枚举`enum` 类型来定义网络状态:

```swift
public enum NetworkReachabilityStatus {
    /// 不确定网络状态
    case unknown
    /// 无法连接
    case notReachable
    /// 可以连接，具体类型由ConnectionType定义
    case reachable(ConnectionType)
    init(_ flags: SCNetworkReachabilityFlags) {
        /// isActuallyReachable 为 Alamofire 定义的 extension 属性
        guard flags.isActuallyReachable else { self = .notReachable; return }
		    /// 先初始化为 .ethernetOrWiFi ，然后判断是否为蜂窝网络，如果是蜂窝网络再转为 . cellular
        var networkStatus: NetworkReachabilityStatus = .reachable(.ethernetOrWiFi)
        if flags.isCellular { networkStatus = .reachable(.cellular) }
        self = networkStatus
    }
    /// 定义可以连接时的网络状态类型
    public enum ConnectionType {
        case ethernetOrWiFi
        case cellular
    }
}
```

### 属性

```swift
/// 初始化单例
public static let `default` = NetworkReachabilityManager()
/// 当前的网络状态是否可以连接
open var isReachable: Bool { return isReachableOnCellular || isReachableOnEthernetOrWiFi }
/// 当前的蜂窝网络是否可用
/// - 用于判断一些是否推荐执行一些需要高或者低带宽的请求，如视频自动播放
open var isReachableOnCellular: Bool { return status == .reachable(.cellular) }
/// 当前的 WiFi 是否可用
open var isReachableOnEthernetOrWiFi: Bool { return status == .reachable(.ethernetOrWiFi) }
/// `DispatchQueue` 系统 Reachability的更新 queue.
public let reachabilityQueue = DispatchQueue(label: "org.alamofire.reachabilityQueue")
/// 可选的 SCNetworkReachabilityFlags
open var flags: SCNetworkReachabilityFlags? {
    var flags = SCNetworkReachabilityFlags()
    return (SCNetworkReachabilityGetFlags(reachability, &flags)) ? flags : nil
}
/// 当前网络状态，使用 Optional 的 map 方法动态生成
open var status: NetworkReachabilityStatus {
    return flags.map(NetworkReachabilityStatus.init) ?? .unknown
}
/// 系统的 `SCNetworkReachability` ，用于发送通知
private let reachability: SCNetworkReachability
/// 线程安全的 `MutableState`
@Protected
private var mutableState = MutableState()
```

### MutableState

`MutableState`为`NetworkReachabilityManager`一些可变状态的封装，`NetworkReachabilityManager`的`mutableState`属性使用 `@Protected`进行声明，保证线程安全。

```swift
struct MutableState {
  	/// 当网络连接状态改变时执行的闭包
    var listener: Listener?
  	/// DispatchQueue监听器将被调用
    var listenerQueue: DispatchQueue?
  	/// 记录前一次网络状态
    var previousStatus: NetworkReachabilityStatus?
}
```

### 网络监听流程

`NetworkReachabilityManager` 提供了两个 `convenience init?` 方法：

```swift
/// 监听指定 host 
public convenience init?(host: String) {
    guard let reachability = SCNetworkReachabilityCreateWithName(nil, host) else { return nil }
    self.init(reachability: reachability)
}
/// 监听 0.0.0.0 地址
public convenience init?() {
    var zero = sockaddr()
    zero.sa_len = UInt8(MemoryLayout<sockaddr>.size)
    zero.sa_family = sa_family_t(AF_INET)
    guard let reachability = SCNetworkReachabilityCreateWithAddress(nil, &zero) else { return nil }
    self.init(reachability: reachability)
}
/// 上面两个方法如果成功生成SCNetworkReachability，到最后会调用这个方法来进行初始化
private init(reachability: SCNetworkReachability) {
    self.reachability = reachability
}
```

`startListening` 添加监听:

```swift
@discardableResult
open func startListening(onQueue queue: DispatchQueue = .main,
                         onUpdatePerforming listener: @escaping Listener) -> Bool {
    stopListening()
    /// 修改 mutableState，这里使用的是@propertyWrapper的语法糖，直接访问projectedValue也就是 Protected<MutableState>
    /// 这样就可以调用 Protected<MutableState> 的 write 方法
    $mutableState.write { state in
        state.listenerQueue = queue
        state.listener = listener
    }
    /// 与SCNetworkReachability交互，添加callBack 。
    var context = SCNetworkReachabilityContext(version: 0,
                                               info: Unmanaged.passUnretained(self).toOpaque(),
                                               retain: nil,
                                               release: nil,
                                               copyDescription: nil)
    let callback: SCNetworkReachabilityCallBack = { _, flags, info in
        guard let info = info else { return }
        let instance = Unmanaged<NetworkReachabilityManager>.fromOpaque(info).takeUnretainedValue()
        instance.notifyListener(flags)
    }
    let queueAdded = SCNetworkReachabilitySetDispatchQueue(reachability, reachabilityQueue)
    let callbackAdded = SCNetworkReachabilitySetCallback(reachability, callback, &context)
    /// 因为SCNetworkReachability不会在初始化时调用一次callBack，所以这里有手动调一下。
    if let currentFlags = flags {
        reachabilityQueue.async {
            self.notifyListener(currentFlags)
        }
    }
    return callbackAdded && queueAdded
}
```

由于`NetworkReachabilityManager.default`获取的是单例，调用`startListening`会清空之前的`mutableState`，导致无法调用之前的`listener`，所以还是使用自己生成的`NetworkReachabilityManager`比较好，各自管理自己的`mutableState` 。

`notifyListener`网络状态变化时调用，执行对应的`listener`:

```swift
func notifyListener(_ flags: SCNetworkReachabilityFlags) {
    let newStatus = NetworkReachabilityStatus(flags)
    $mutableState.write { state in
		  	/// 如果状态相等就不调用 listener 
        guard state.previousStatus != newStatus else { return }
        state.previousStatus = newStatus
        let listener = state.listener
        state.listenerQueue?.async { listener?(newStatus) }
    }
}
```

`stopListening`停止监听，调用`SCNetworkReachabilitySetCallback`和`SCNetworkReachabilitySetDispatchQueue`清空 `reachability`，然后清空`mutableState`的属性:

```swift
/// 停止监听网络连接状态的变化
open func stopListening() {
    SCNetworkReachabilitySetCallback(reachability, nil, nil)
    SCNetworkReachabilitySetDispatchQueue(reachability, nil)
    $mutableState.write { state in
        state.listener = nil
        state.listenerQueue = nil
        state.previousStatus = nil
    }
}
```

### 总结

+ 当网络状态切换时，可以考虑重新发送请求
+ 网络状态可以为用户请求失败提供一些更明确的提示
+ 判断用户的网络状态来执行不同的流量策略，如视频的清晰度
+ 不要通过 `NetworkReachabilityManager` 来判断是否应该发送请求