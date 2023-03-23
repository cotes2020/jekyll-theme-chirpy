---
title: HTTP - Retry
# author: Grace JyL
date: 2021-04-25 11:11:11 -0400
description:
excerpt_separator:
categories: [Web, HTTP]
tags: [Web, HTTP]
math: true
# pin: true
toc: true
# image: /assets/img/sample/devices-mockup.png
---

- [HTTP - Retry](#http---retry)
  - [什么情况下要 retry](#什么情况下要-retry)
  - [Retry处理步骤](#retry处理步骤)
    - [**简单的立即retry**](#简单的立即retry)
    - [**有延迟的retry**](#有延迟的retry)
      - [固定间隔的 delay](#固定间隔的-delay)
      - [随机 delay 的方式：](#随机-delay-的方式)
    - [**Circuit Breaker(断路器)**](#circuit-breaker断路器)
  - [Retry 设计模式在客户端的应用与实现](#retry-设计模式在客户端的应用与实现)
  - [监控：更好地在运行时了解你的系统](#监控更好地在运行时了解你的系统)
  - [Example](#example)
    - [go-resty 重试机制的实现](#go-resty-重试机制的实现)
    - [Backoff函数](#backoff函数)
    - [Demo](#demo)
    - [一些其他重试机制的实现](#一些其他重试机制的实现)

- ref
  - [浅谈Retry设计模式及在前端的应用与实现](https://segmentfault.com/a/1190000022418493)
  - [重试机制的实现](https://segmentfault.com/a/1190000025181043)
  - [如何正确地实现重试 (Retry)](https://www.infoq.cn/article/7z0wpahuh9euxqiq5xzx)



---

# HTTP - Retry

---


## 什么情况下要 retry

认识 Transient fault（短暂故障）
- 短暂存在，并且在一段时间后会被修复的故障。
- 这个时间可以短到几毫秒也可以长达几小时。
- 如果的请求是因为一个这样的故障而失败的，那在适当的时候重试就可以了。

> 在前端应用中，短暂故障往往发生在你向服务端请求资源时。比如你向一个API发送一个AJAX请求，对面返回一个 “5XX” 的响应。



鉴别 Transient fault（短暂故障）
- 造成请求失败的原因有很多。
  - 因为服务端内部逻辑的错误，
  - 客户端发送了一个糟糕的请求，
  - 由Infrastructure造成的一些短暂故障（比如暂时负载量过大，某个服务器的宕机，或网络问题等）。
- 而只有在短暂故障的情况下进行retry才有意义。
- 鉴别短暂故障
  - 最简单的方法是运用 HTTP 请求的响应码。根据规范，
  - 400-499 之间是客户端造成的问题，没有必要重试了
  - 500-599 之间是服务端的故障
    - 如何在5xx 的故障中鉴别出短暂故障。
    - 如果服务端对错误响应码有标准的定义，就可以通过不同的号码得知错误的原因，从而决定是进行retry还是做别的处理。
    - 服务端开发中标准并清晰的定义错误码和给与错误信息的重要性。

---

## Retry处理步骤

当请求失败时可以有一个基本的处理步骤：
- 鉴别是不是 transient fault
- 如果是，启动retry机制，进行一定次数的retry
- 当retry达到最大次数还没有成功，报错并说明原因：服务端暂时无法响应。


基本的retry设计模式。

### **简单的立即retry**
- 当请求失败，立即retry，
- 用于一些不常见的失败原因，因为原因罕见，立刻retry也许就修复了。
- 但当碰到一些常见的失败原因如服务端负载过高，不断的立即retry只会让服务端更加不堪重负。
- 试想如果有多个客户端instance在同时发送请求，那越是retry情况就越糟糕。
- **不带 backoff 的重试，对于下游来说会在失败发生时进一步遇到更多的请求压力，进一步恶化**。

```java
public static <T> T retryNoDelay(final callable<T> callable, final int maxAttempts){
  for (int i = 0; i < maxAttempts; i++) {
    try {
      final T t = callable.call();
      if (isExpected(t)){ return t};
    }
    try {
      insertMessageInboxResult = messageInboxManager.insertMessage(messageInbox);
      if (insertMessageInboxResult) {
        break;
      }
    }
    catch (Exception e) {
      log.error(
        "insertMessageInbox exception retry {}, messageInbox={}", i, messageInbox)
    }
  }
  // return default t or error
  return null;
}
```

---

### **有延迟的retry**
- 与其立即retry, 倒不如等待一会，也许那时服务端的负载就降下来了。
- 这个 delay（延迟）的时间可以是一个常量，也可以是根据一定数学公式变化的变量。
  - 逐次增加delay算法。
  - Exponential Backoff (指数后退算法): 以指数的方式来增加delay。
    - such as 第一次失败等待1秒，第二次再失败等待2秒，接下去4秒，8秒...。

> 根据自己系统的特性和业务的需求，设计更适合更优化的算法。


#### 固定间隔的 delay

- delay 的方式按照是方法本身是异步还是同步的，可以通过定时器或则简单的 `Thread.sleep` 实现

- 虽然这次带了固定间隔的 backoff，但是**每次重试的间隔固定**
- 此时对于下游资源的冲击将会变成间歇性的脉冲；
- 特别是当集群都遇到类似的问题时，步调一致的脉冲，将会最终对资源造成很大的冲击，并陷入失败的循环中。

```java
public static <T> T retry(
  final Callable<T> callable, final int maxAttempts, final int fixedBackOff){
    for (int i = 0; i < maxAttempts; i++) {
      try {
        final T t = callable.call();
        if (isExpected(t)){ return t};
      }
      catch (Exception e) {
        log.error("error")
        try {
          Thread.sleep(fixedBackOff);
        }
        catch (Exception ee) {}
      }
    }
    // return default t or error
    return null;
  }
```


```java
private T retry(Supplier<T> function) throws RuntimeException {
    log.error("1st command failed, will be retired " + maxRetires + "times.");
    invokeCnt = 0;
    Exception exception = null;

    while ( invokeCnt < maxRetires) {
      try {
        if (sleepTime > MIN_SLEEP_TIME && sleepTime <= MAX_SLEEP_TIME) {
          Thread,sleep(sleepTime);
        }
        return function.get();
      }
      catch (InterrunptedException ex) {...}
      catch (Exception ex) {
        exception = ex;
        invokeCnt++;
        log.error(invokeCnt + " times retur failed of " + maxRetires + " error: " + ex);
        if ( invokeCnt > maxRetires) {...}
      }
    }
    throw new RuntimeException(maxRetires + " retires all failes", exception);
  }
```


#### 随机 delay 的方式：

- 采用随机 backoff 的方式，即具体的 delay 时间在一个最小值和最大值之间浮动
- 虽然解决了 backoff 的时间集中的问题，对时间进行了随机打散，但是依然存在下面的问题：
- 如果依赖的底层服务持续地失败，改方法依然会进行固定次数的尝试，并不能起到很好的保护作用
- 对结果是否符合预期，是否需要进行重试依赖于异常
- 无法针对异常进行精细化的控制，如只针部分异常进行重试。


```java

public static <T> T retryWithRandomDelay(
  final Callable<T> callable,
  final int maxAttempts,
  final int minBackOff,
  final int maxBackOff,
  // randomFactor, 0.0 - 1.0
  final double randomFactor) {
    for (int i = 0; i < maxAttempts; i++) {
      try {
        final T t = callable.call();
        if (isExpected(t)) {
          return t
        };
      }
      catch (Exception e) {
        log.error("error")
        try {
          final double rnd = 1.0 + ThreadLocalRandom.current().nextDouble() * randomFactor;
          long backOffTime;
          try {
            backOffTime = (long)(Math.min(maxBackoff, minBackoff * Math.pow(2, i)) * rnd);
          }
          catch (Exception ee) {
            backOffTime = maxBackoff;
          }
          Thread.sleep(backOffTime);
        }
        catch (Exception ee) {}
      }
    }
    // return default t or error
    return null;
  }
```


```java
public static <V> V retryWithRandomDelay(
  final Callable<V> callable,
  final int maxRetryTime,
  final int sleepInMills) throws Exception {
    if (maxRetryTime >= 10) {...}
    if (sleepInMills <= 0) {...}

    try {
      return callable.call();
    }
    catch (Throwable e) {
      if (maxRetryTime <= 0) {...}
      else {
        try {
          LOGGGER.error(
            "retry with maxRetryTime:{}, sleepInMills:{}, error:{}",
            maxRetryTime,sleepInMills,e
          )
          Thread.sleep(sleepInMills);
          return retryWithBackoff(
            callable,
            maxRetryTime: maxRetryTime-1,
            sleepInMills: sleepInMills * ThreadLocalRandom.current().nextDouble(origin:1, bound: 3));
        }
        catch (InterrunptedException ex) {
          Thread.currentThread().interrupt();
          throw e;
        }
      }
    }
  }
```



---


### **Circuit Breaker(断路器)**
- 如果 Transient fault 修复的时间特别长, 比如长时间的网络问题，那就算有再好的retry机制，也免不了是徒劳。只会一次又一次地retry, 失败，再retry, 直到达到上限。
  - 一来浪费资源，二来或许又会干扰服务端的自我修复。
  - 断路器模式 一般用在当下游资源失败后，但是失败恢复的时间不固定时，自动地进行探索式地恢复尝试，并且在遇到较多失败时，能够快速自动地断开，从而避免失败蔓延的一种模式。
  - 当断路器处于开断状态时，所有的请求都会直接失败，不再会对下游资源造成冲击，并能够在一段时间后，进行探索式的尝试，如果没有达到条件，可以自动地恢复到之前的闭合状态。

- Circuit Breaker (断路器)的设计模式: 原意其实就是电路中的开关
    - 在电路里一旦开关断开，电流就别想通过了。
    - 一旦开关断开，就不会再发送任何请求了。


- Circuit Breaker在retry机制中的应用是一个状态机
  - 有三种状态：OPEN,HALF-OPEN, CLOSE。
  - 设定一个 `threshold(阈值)` 和一个 `timeout`，
  - 当retry的次数超过 `threshold` 时，认为服务端进入一个较长的Trasient fault。
  - 那么就开关断开，进入 **OPEN** 状态。这时将不再向服务端发送任何请求，就像开关断开了，电流（请求）怎么也不会从客户端流向服务端。
  - 当过了一段时间到了 `timeout`，就将状态设为 **HALF-OPEN**，这时会尝试把客户端的请求发往服务端去试探它是否已经恢复。
  - 如果是就进入 **CLOSE** 状态，回到正常的机制中
  - 如果不是，就再次进入 **OPEN** 状态。

> 既节约了资源，防止了无休止的无用的尝试，
> 又保证了在修复后，客户端能知晓，并恢复的正常的运行。

![Screen Shot 2021-09-02 at 01.37.21](https://i.imgur.com/I2d9u6d.png)

在应用断路器时，需要对下游资源的每次调用都通过断路器，对代码具备一定的结构侵入性。常见的有 Hystrix 或 resilience4j.

```java
// Given
CircuitBreaker circuitBreaker = CircuitBreaker.ofDefaults("testName");

// When I decorate my function
CheckedFunction0<String> decoratedSupplier = CircuitBreaker
        .decorateCheckedSupplier(circuitBreaker, () -> "This can be any method which returns: 'Hello");

// 又或者
def callWithCircuitBreakerCS[](body: Callable[CompletionStage[T]]): CompletionStage[T]

```











---

## Retry 设计模式在客户端的应用与实现

- 在服务端，Retry 的机制被大量运用，尤其是在云端微服务的架构上。很多云平台本身就提供了主体（比如服务，节点等）之间的retry机制从而提高整个系统的稳定性。

- 而客户端，作为一个独立于服务端系统之外，运行在用户手机或电脑上的一个App, 并没有办法享受到平台的这个功能。
  - 这时，就需要为App加入retry机制, 从而使整个产品更加强壮。

npm 有一个 retry 的包可以帮助我们快速加入retry机制: https://www.npmjs.com/package

retry的实现并不复杂
- 完全可以自己写一个这样的工具供一个或多个产品使用。
- 更容易更改其中的算法来适应产品的需求。

下面是我写的一个简单的retry小工具，由于我们向服务端做请求的函数常常是返回promise的，比如 fetch 函数 。这个工具可以为任何一个返回promise的函数注入retry机制。

```java
// 这个函数会为你的 promiseFunction (一个返回promise的函数) 注入retry的机制。
// 比如 retryPromiseFunctionGenerator(myPromiseFunction, 4, 1000, true, 4000)
// 会返回一个函数，它的用法和功能与 myPromiseFunction 一样。但如果 Promise reject 了，
// 它就会进行retry, 最多retry 4 次
// 每次时间间隔指数增加，最初是1秒，随后2秒，4秒，
// 由于设定最大delay是4秒，那么之后就会持续delay4秒，直到达到最大retry次数 4 次。
// 如果 enableExponentialBackoff 设为 false, delay就会是一个常量1秒。
const retryPromiseFunctionGenerator = (
  promiseFunction, // 需要被retry的function
  numRetries = defaultNumRetries, // 最多retry几次
  retryDelayMs = defaultRetryDelayMs, // 两次retry间的delay
  enableExponentialBackoff = false, // 是否启动指数增加delay
  maxRetryDelayMs // 最大delay时间
) => async (...args) => {
  for (
    let numRetriesLeft = numRetries;
    numRetriesLeft >= 0;
    numRetriesLeft -= 1
  ) {
    try {
      return await promiseFunction(...args);
    } catch (error) {
      if (numRetriesLeft === 0 || !isTransientFault(error)) {
        throw error;
      }

      const delay = enableExponentialBackoff
        ? Math.min(
            retryDelayMs * 2 ** (numRetries - numRetriesLeft),
            maxRetryDelayMs || Number.MAX_SAFE_INTEGER
          )
        : retryDelayMs;

      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
};
```

---


## 监控：更好地在运行时了解你的系统

App拥有了retry机制，在客户端运行时，它变得更强壮了，一些失败的服务端请求并不能打到它。

但想知道它在用户手上retry了几次，什么时候retry的，最终失败了没有。
- 这些信息不仅让我更好的了解用户的实际体验，它们也可以作为服务端性能的指标之一。
- 实时对这些信息进行监控可以**尽早的发现服务端的故障** 以减少损失

客户端的监控是一个很大的话题，Retry信息的收集只是其中一个应用场景。
- 实现呢很简单。
- 在每一次执行retry时发送一条log(日志)其中包含你想了解的信息。然后运用第三方或公司内部的日志浏览工具去分析这些日志，从而获得许多有意思的指标。
- 例子，我们可以简单地监控retry log 的数量，如果突然激增，那就说明服务端也许出现了一些故障，这时候开发团队可以在第一时间做出反应修复故障，以免对大面积的客户造成影响。
- 当然这不仅仅可以通过监控retry实现，我们也可以监控服务端的http请求失败的数量，


---


## Example

- 服务在请求资源，如果遇到网络异常等情况，导致请求失败，这时需要有个重试机制来继续请求。
- 常见的做法是重试3次，并随机 sleep 几秒。
- 业务开发的脚手架，HTTP Client 基本会封装好 retry 方法，请求失败时根据配置自动重试。
- 下面以一个常见的 HTTP Client 为例， 看下它是如何实现请求重试。
- 最后整理其他一些重试机制的实现。


### go-resty 重试机制的实现
先看下 go-resty 在发送 HTTP 请求时， 请求重试的实现：

```go
// Execute method performs the HTTP request with given HTTP method and URL

// for current `Request`.
//   resp, err := client.R().Execute(resty.GET, "http://httpbin.org/get")
func (r *Request) Execute(method, url string) (*Response, error) {
    var addrs []*net.SRV
    var resp *Response
    var err error

    if r.isMultiPart && !(method == MethodPost || method == MethodPut || method == MethodPatch) {
        return nil, fmt.Errorf("multipart content is not allowed in HTTP verb [%v]", method)
    }

    if r.SRV != nil {
        _, addrs, err = net.LookupSRV(r.SRV.Service, "tcp", r.SRV.Domain)
        if err != nil {
            return nil, err
        }
    }

    r.Method = method
    r.URL = r.selectAddr(addrs, url, 0)

    if r.client.RetryCount == 0 {
        resp, err = r.client.execute(r)
        return resp, unwrapNoRetryErr(err)
    }
    // 如果 r.client.RetryCount 不等于0 ，执行 Backoff() 函数

    // Backoff() 方法接收一个处理函数参数
    // 根据重试策略，进行 attempt 次网络请求，同时接收 Retries()、WaitTime()等函数参数
    attempt := 0
    err = Backoff(
        func() (*Response, error) {
            attempt++

            r.URL = r.selectAddr(addrs, url, attempt)

            resp, err = r.client.execute(r)
            // 如果没有设置重试次数，执行 r.client.execute(r) ：
            // 直接请求 Request ， 返回 Response 和 error。

            if err != nil {
                r.client.log.Errorf("%v, Attempt %v", err, attempt)
            }

            return resp, err
        },
        Retries(r.client.RetryCount),
        WaitTime(r.client.RetryWaitTime),
        MaxWaitTime(r.client.RetryMaxWaitTime),
        RetryConditions(r.client.RetryConditions),
    )
    return resp, unwrapNoRetryErr(err)
}
```

### Backoff函数

```go
// Backoff retries with increasing timeout duration up until X amount of retries
// (Default is 3 attempts, Override with option Retries(n))
func Backoff(operation func() (*Response, error), options ...Option) error {
    // Defaults
    opts := Options{
        maxRetries:      defaultMaxRetries,
        waitTime:        defaultWaitTime,
        maxWaitTime:     defaultMaxWaitTime,
        retryConditions: []RetryConditionFunc{},
    }

    for _, o := range options {
        o(&opts)
    }

    var (
        resp *Response
        err  error
    )

    // 开始进行 opts.maxRetries 次 HTTP 请求
    for attempt := 0; attempt <= opts.maxRetries; attempt++ {
        // 执行处理函数 (发起 HTTP 请求)
        resp, err = operation()
        ctx := context.Background()

        // 如果返回结果不为空并且 context 不为空，
        // 保持 repsonse 的请求上下文。
        if resp != nil && resp.Request.ctx != nil {
            ctx = resp.Request.ctx
        }
        // 如果上下文出错， 退出 Backoff() 流程
        if ctx.Err() != nil {
            return err
        }

        err1 := unwrapNoRetryErr(err)
        // raw error, it used for return users callback.

        needsRetry := err != nil && err == err1
        // retry on a few operation errors by default

        // 执行 retryConditions(), 设置检查重试的条件。
        for _, condition := range opts.retryConditions {
            needsRetry = condition(resp, err1)
            if needsRetry {
                break
            }
        }

        if !needsRetry { //根据 needsRetry 判断是否退出流程
            return err
        }

        // 通过 sleepDuration()计算 Duration
        // （根据此次请求resp, 等待时间配置，最大超时时间和重试次数算出 sleepDuration。
        // 时间算法相对复杂， 具体参考： Exponential Backoff And Jitter）
        waitTime, err2 := sleepDuration(resp, opts.waitTime, opts.maxWaitTime, attempt)
        if err2 != nil {
            if err == nil {
                err = err2
            }
            return err
        }

        // 等待 waitTime 进行下个重试。 如果请求完成退出流程
        select {
        case <-time.After(waitTime):
        case <-ctx.Done():
            return ctx.Err()
        }
    }

    return err
}
```

### Demo

看具体 HTTP Client （有做过简单封装）的请求:

```go
func getInfo() {
  request := client.DefaultClient().NewRestyRequest(
    ctx, "", client.RequestOptions{
      MaxTries:      3,
      RetryWaitTime: 500 * time.Millisecond,
      RetryConditionFunc: func(response *resty.Response) (b bool, err error) {
        if !response.IsSuccess() { return true, nil }
        return
      },
    }).SetAuthToken(args.Token)

    // 然后 request.Get(url) 进入到 Backoff() 流程，
    // 此时重试的边界条件是： !response.IsSuccess(), 直到请求成功。
    resp, err := request.Get(url)

    if err != nil {
        logger.Error(ctx, err)
    return
    }

    body := resp.Body()
    if resp.StatusCode() != 200 {
    logger.Error(
      ctx, fmt.Sprintf("Request keycloak access token failed, messages:%s, body:%s","message", resp.Status(),string(body))),
        )
    return
    }
  ...
}
```


---


### 一些其他重试机制的实现

可以看出其实 go-resty 的 重试策略不是很简单， 这是一个完善，可定制化， 充分考虑 HTTP 请求场景下的一个机制， 它的业务属性相对比较重。


实现一

每次重试等待随机延长的时间， 直到 f() 执行完成 或不再重试。

```go
// retry retries ephemeral errors from f up to an arbitrary timeout
func retry(f func() (err error, mayRetry bool)) error {
    var (
        bestErr     error
        lowestErrno syscall.Errno
        start       time.Time
        nextSleep   time.Duration = 1 * time.Millisecond
    )

    for {
        err, mayRetry := f()
        if err == nil || !mayRetry {
            return err
        }

        if errno, ok := err.(syscall.Errno); ok && (lowestErrno == 0 || errno < lowestErrno) {
            bestErr = err
            lowestErrno = errno
        } else if bestErr == nil {
            bestErr = err
        }

        if start.IsZero() {
            start = time.Now()
        } else if d := time.Since(start) + nextSleep; d >= arbitraryTimeout {
            break
        }
        time.Sleep(nextSleep)
        nextSleep += time.Duration(rand.Int63n(int64(nextSleep)))
    }

    return bestErr
}
```


实现二

对函数重试 attempts 次，每次等待 sleep 时间， 直到 f() 执行完成。


````go
func Retry(attempts int, sleep time.Duration, f func() error) (err error) {
    for i := 0; ; i++ {
        err = f()
        if err == nil {
            return
        }

        if i >= (attempts - 1) {
            break
        }

        time.Sleep(sleep)

    }
    return fmt.Errorf("after %d attempts, last error: %v", attempts, err)
}
```
















.
