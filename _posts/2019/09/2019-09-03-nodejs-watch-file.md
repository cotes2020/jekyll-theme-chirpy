---
title: "NodeJS 如何监听文件变化？"
date: 2019-09-03
permalink: /2019-09-03-nodejs-watch-file/
---
Keywords: 操作系统差异、识别用户/编辑器操作、连续触发的优化、工程级 API。


## 概述


NodeJS 提供了 fs.watch / fs.watchFile 两种 API：

- fs.watch: 推荐，可以监听文件夹。基于操作系统。
- fs.watchFile: 只能监听指定文件。并且通过轮询检测文件变化，不能响应实时反馈。

一个监听指定文件夹的代码如下：


```typescript
fs.watch(dir, { recursive: true }, (eventType, file) => {
    if (file && eventType === "change") {
        console.log(`${file} 已经改变`);
    }
})
```


## 跨平台优化


> 对于不同系统内核，比如 maxos，fs.watch 回调函数中的第一个参数，不会监听到 rename、delete 事件。因此，这不是一个工程级别的可用 api。


### 文件 md5


某些开源软件，会将文件内容都清空后，再添加内容。而且保存过程中，可能会出现多个中间态。


对于文件更改的情况，检测内容的 md5 值，是个不错的方法。


```typescript
let previousMD5 = "";
fs.watch("./whatever", (type, filename) => {
    if (!filename) {
        return;
    }
    const md5 = crypto.createHash("md5");
    const currentMD5 = md5
        .update(fs.readFileSync(filename).toString())
        .digest("hex");
    if (currentMD5 === previousMD5) {
        return;
    }
    previousMD5 = currentMD5;
    console.log(`${filename} is changed`);
});
```


### 事件频率控制


对于文件变更，不同的系统可能会触发多个不同的中间态。因此，借助 debounce 函数的思想，控制和修正回调事件的触发频率。


前面的代码修正为：


```typescript
let previousMD5 = "";
let watchWait = false; //
fs.watch("./whatever", (type, filename) => {
    if (!filename || watchWait) {
        return;
    }
    //
    watchWait = setTimeout(() => {
        watchWait = false;
    }, 100);
    const md5 = crypto.createHash("md5");
    const currentMD5 = md5
        .update(fs.readFileSync(filename).toString())
        .digest("hex");
    if (currentMD5 === previousMD5) {
        return;
    }
    previousMD5 = currentMD5;
    console.log(`${filename} is changed`);
})
```


### 文件信息


对于常见的库来说，除了不信任原生 API、使用上述技巧外，很重要的是，**都根据 fs.Stats 类的信息，自定义逻辑来判断文件状态，以此保证不同平台兼容性**。


下面是在 Node10 中，打印的文件状态信息：


```shell
Stats {
  dev: 16777222,
  mode: 33188,
  nlink: 1,
  uid: 501,
  gid: 20,
  rdev: 0,
  blksize: 4096,
  ino: 6493141,
  size: 7,
  blocks: 8,
  atimeMs: 1567516873292.676,
  mtimeMs: 1567516873293.3867,
  ctimeMs: 1567516873293.3867,
  birthtimeMs: 1566547653640.1763,
  atime: 2019-09-03T13:21:13.293Z,
  mtime: 2019-09-03T13:21:13.293Z,
  ctime: 2019-09-03T13:21:13.293Z,
  birthtime: 2019-08-23T08:07:33.640Z }
```


通过文件信息的思路，就是在`fs.stat()`的回调函数中，进行逻辑处理：


```typescript
// 判断文件是否写入完毕的操作
function awaitWriteFinish() {
    // ...省略
    fs.stat(
        fullPath,
        function(err, curStat) {
            // ...省略
            if (prevStat && curStat.size != prevStat.size) {
                this._pendingWrites[path].lastChange = now;
            }
            if (now - this._pendingWrites[path].lastChange >= threshold) {
                delete this._pendingWrites[path];
                awfEmit(null, curStat);
            } else {
                timeoutHandler = setTimeout(
                    awaitWriteFinish.bind(this, curStat),
                    this.options.awaitWriteFinish.pollInterval
                );
            }
        }.bind(this)
    );
    // ...省略
}
```


## 成熟的库

- [nodemon](https://github.com/remy/nodemon/)

## 参考链接

- [精读《如何利用 Nodejs 监听文件夹》](https://segmentfault.com/a/1190000015159683)

