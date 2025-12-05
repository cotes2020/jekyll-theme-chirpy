---
title: "NodeJS 服务错误处理经验"
url: "2022-01-25-node-server-err"
date: 2022-01-25
---

## 如何在KoaJS中处理错误？


KoaJS是洋葱模型，请求进入和返回时，会“经过”2次中间件。所以可以直接在第一个中间件中，通过try-catch捕获服务的错误。


```javascript
async function errHandlerMiddleware(ctx, next) {
  logger.log({
    logType: 'accessLog',
    reqPath: ctx.request.path,
    reqId: ctx.request.reqId
    // ... 更多信息
  })

	try {
		await next()
	} catch (err) {
		logger.error({
      logType: 'errHandle',
      reqId: ctx.request.reqId,
      errStack: inspect(err.stack),
      errMsg: err.message
    })
		ctx.res.status = 500
		ctx.res.body = "server error"
	}
}
```


配合process.on，可以监听未捕获的`unhandledRejection`、`uncaughtException`


## 如何在Express中处理错误？


Experss不是洋葱模型，而是中间件进行「串行处理」，请求返回时，不会再“经过”前面的中间件。所以，要在中间件中，通过事件监听，来响应服务中的报错，并且进行打印：


```javascript
function errHandlerMiddleware(req, res, next) {
	res.on('finish', () => {
		// ...
	})
	res.on('error', (err) => {
		// ...
	})
}
```


> 在这个点上，express 不如 koa 优雅。


配合process.on，可以监听未捕获的`unhandledRejection`、`uncaughtException`


## 如何在NestJS中处理错误？


### 分类讨论错误类型


由于Nest基于Express 2次开发，并且加了Ioc，因此处理起来比较棘手。


按照错误类型，可以按照以下方法来处理：

1. 未捕获的`unhandledRejection`、`uncaughtException`，比如异步处理报错、第三方库错误：通过process.on监听
2. NestJS内置错误 `HttpException` ：通过全局Filter来捕获
3. 服务定义的标准错误，继承自 `HttpException` ：通过全局Filter来捕获
4. 服务未定义的错误：通过全局Filter来捕获
5. 第三方库的错误：通过全局Filter来捕获
	1. HTTP 库的错误
	2. 非 HTTP 错误

### 实现标准错误类


项目中不允许直接在代码中，直接`throw new Errro(xxxx)`抛出错误。此类错误会被Filter识别为`sysErr`系统错误。


所有的错误都要通过标准的方法`makeErr`进行抛出。


抛出的错误继承自 `HttpException`，新增了错误码`code`属性。


整体实现如下：


```typescript
import { HttpException, HttpStatus } from '@nestjs/common';
import { isString } from 'lodash';

interface IMakeErrOpts {
  code: string;
  message: string;
  statusCode?: number;
  stack?: string;
}

/**
 * 错误基类
 */
export class BaseError extends HttpException {
  /**
   * 业务错误码
   */
  public readonly code: string;

  /**
   * 错误基类
   * @param code 业务错误码
   * @param message 错误信息
   * @param stack 错误堆栈
   * @param statusCode HTTP错误码
   */
  constructor(code, message, stack, statusCode) {
    super(message, statusCode);
    if (isString(stack)) {
      // 默认使用父类的上下文堆栈
      this.stack = stack;
    }
    this.code = code;
  }
}

/**
 * 生成标准错误
 * @param opts
 * @returns
 */
export function makeErr(opts: IMakeErrOpts) {
  const {
    code,
    message,
    stack,
    statusCode = HttpStatus.INTERNAL_SERVER_ERROR,
  } = opts;
  return new BaseError(code, message, stack, statusCode);
}

```


### 实现错误捕获Filter


从上面看到，2-5都通过Filter来捕获。因此，要在Filter中区分处理不同类型报错。

- 内置错误和继承自内置错误的标准错误：返回业务的错误码，打印错误信息
- 未定义错误：开发阶段未发现、未处理成标准错误的异常。返回系统错误码，红体打印系统错误。
- 第三方库的错误：
	- HTTP 库错误：error上一般都有status，识别status即可。如果status是4xx，那么处理和第一类一样；否则，和第二类一样
	- 非 HTTP 错误：看是否提供事件监听或者try-catch捕获，不然就是和第二类一样，属于未捕获错误。

整体实现如下：


```typescript
@Catch()
export class AllExceptionFilter implements ExceptionFilter {
  catch(exception: any, host: ArgumentsHost) {
    const ctx = host.switchToHttp();
    const res: ICtxResponse = ctx.getResponse();
    const req: ICtxRequest = ctx.getRequest();
    const isSysErr = !(
      exception instanceof HttpException ||
      (exception.status >= 400 && exception.status < 500)
    ); // 正常错误包括：nest内置错误、继承nest内置错误的标准错误、第三方http库的4xx系列错误

    // 日志类型
    const logType = isSysErr ? 'sysErr' : 'responseErr';
    // HTTP 状态码
    const status: number = isSysErr
      ? HttpStatus.INTERNAL_SERVER_ERROR
      : exception.status || HttpStatus.BAD_REQUEST;
    // 业务错误码。用户可根据错误码来细化前端交互
    const errCode: string = isSysErr
      ? SERVER_ERR_CODE
      : exception.code || BAD_REQUEST_CODE;
    // 错误详情
    const errMsg: string = exception?.response?.message
      ? exception.message + '. ' + exception?.response?.message
      : exception.message;

    // 额外打印错误信息
    baseLog.print({
      logLevel: isSysErr ? 'error' : 'warn',
      reqId: req.reqId,
      logType,
      logTime: Date.now(),
      errStatus: status,
      errMsg,
      errStack: exception.stack,
      errCode,
    });

    // 用于 accessLog 日志打印
    req.accessErr = {
      status,
      code: errCode,
      message: errMsg,
    };

    // 返回给用户标准数据
    res.set('X-Content-Type-Options', 'nosniff').status(200).json({
      reqId: req.reqId,
      code: errCode,
      msg: errMsg,
    });
  }
}
```


